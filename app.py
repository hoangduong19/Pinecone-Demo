import streamlit as st
import torch
import time
import random
import numpy as np
from datetime import datetime, timedelta
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from pinecone import Pinecone, ServerlessSpec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- CONFIGURATION & UI ---
st.set_page_config(page_title="Vector DB Scaling Demo", layout="wide")
st.title("Pinecone Serverless: Scaling & Performance Benchmark")

# --- KHỞI TẠO SESSION STATE CHO SCALING HISTORY ---
if "history" not in st.session_state:
    st.session_state.history = []
if "total_data" not in st.session_state:
    st.session_state.total_data = 0

# --- 1. RESOURCES (Caching & Xáo trộn dữ liệu) ---
@st.cache_resource
def init_resources(data_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ID = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_ID).to(device)
    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    
    ds_fashion_full = load_dataset("ashraq/fashion-product-images-small", split='train')
    ds_fashion = ds_fashion_full.shuffle(seed=42).select(range(data_size))
    
    ds_food_full = load_dataset("RitishaAmod123/food-ingredients-dataset", split='train')
    ds_food = ds_food_full.shuffle(seed=42).select(range(data_size))
    
    return model, processor, tokenizer, ds_fashion, ds_food, device

model, processor, tokenizer, ds_fashion, ds_food, device = init_resources(500)

# --- 2. EMBEDDING & BENCHMARK FUNCTIONS ---
def get_image_embedding(image):
    inputs = processor(text=None, images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
    return embeddings.cpu().numpy().tolist()[0]

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
    return embeddings.cpu().numpy().tolist()[0]

def benchmark_avg(index, q_vec, namespace, n=5):
    times = []
    for _ in range(n):
        start = time.time()
        index.query(vector=q_vec, top_k=5, namespace=namespace)
        times.append((time.time() - start) * 1000)
    return sum(times) / len(times)

def measure_qps(index, q_vec, namespace, duration=2):
    count = 0
    start = time.time()
    while time.time() - start < duration:
        index.query(vector=q_vec, top_k=5, namespace=namespace)
        count += 1
    return count / duration

def calculate_recall(pinecone_results, q_vec, namespace, top_k=4):
    if 'local_db' not in st.session_state or namespace not in st.session_state.local_db:
        return 0.0 
        
    local_data = st.session_state.local_db[namespace]
    if not local_data: return 0.0
    
    all_vecs = [item['vec'] for item in local_data]
    all_ids = [item['id'] for item in local_data]
    
    scores = cosine_similarity([q_vec], all_vecs)[0]
    ground_truth_idx = np.argsort(scores)[-top_k:][::-1]
    ground_truth_ids = [all_ids[i] for i in ground_truth_idx]
    
    pinecone_ids = [m['id'] for m in pinecone_results['matches']]
    matches = len(set(ground_truth_ids).intersection(set(pinecone_ids)))
    return (matches / top_k) * 100

# --- 3. SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Admin & Benchmark")
    api_key = st.secrets["PINECONE_API_KEY"]
    
    index_name = "scaling-demo-v3" 
    
    st.divider()
    st.subheader("📊 Scaling Demo Config")
    data_size = st.selectbox("Dataset size (mỗi Namespace)", [100, 300, 500, 1000, 5000], index=2)
    num_requests = st.slider("Load test (số query)", 1, 1000, 10)
    run_stress = st.checkbox("🔥 Run stress test")

    if st.button("🧹 Xóa Lịch sử Benchmark"):
        st.session_state.history = []
        st.session_state.total_data = 0
        st.rerun()
        
    model, processor, tokenizer, ds_fashion, ds_food, device = init_resources(data_size)

    # --- HAI NÚT RIÊNG BIỆT ---
    st.markdown("### Hành động nạp dữ liệu")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        btn_reset = st.button("🚀 Reset (Cold)", type="primary", use_container_width=True, help="Đập đi xây lại Database mới tinh")
    with col_btn2:
        btn_append = st.button("➕ Nạp Thêm", type="secondary", use_container_width=True, help="Giữ nguyên DB, nạp dồn thêm data")

    if btn_reset or btn_append:
        if not api_key: st.error("Nhập API Key!"); st.stop()
        pc = Pinecone(api_key=api_key)
        
        prov_time = 0.0 # Khởi tạo biến thời gian Provisioning
        
        # 1. NẾU BẤM RESET -> XÓA VÀ TẠO MỚI
        if btn_reset:
            if index_name in pc.list_indexes().names():
                pc.delete_index(index_name)
            
            st.write("⏳ Đang khởi tạo Serverless Index (Cold Start)...")
            prov_start = time.time()
            pc.create_index(name=index_name, dimension=512, metric="cosine", 
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            prov_time = time.time() - prov_start
            st.success(f"✅ Provisioning Time: {prov_time:.2f} giây")
            
            # Reset lại bộ đệm ở Local
            st.session_state.local_db = {"fashion_ns": [], "food_ns": []}
            st.session_state.total_data = 0
            
        # 2. NẾU BẤM NẠP THÊM -> CHỈ KẾT NỐI
        else:
            st.write("⏳ Đang kết nối vào Index hiện tại để nạp thêm (Incremental)...")
            if 'local_db' not in st.session_state:
                st.session_state.local_db = {"fashion_ns": [], "food_ns": []}

        index = pc.Index(index_name)
        start_upsert = time.time()
        
        def process_and_upsert(dataset, namespace, prefix, default_city):
            # Kích thước mẻ xử lý (Vừa đủ để AI tính nhanh, vừa đủ để Pinecone nhận mượt)
            batch_size = 64 
            start_offset = len(st.session_state.local_db[namespace])
            
            # Cắt dataset ra thành từng khúc (chunk), mỗi khúc 64 ảnh
            for batch_start in range(0, len(dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(dataset))
                
                # --- CHÌA KHÓA SỬA LỖI Ở ĐÂY ---
                # Lấy từng phần tử ra đóng gói thành 1 List chuẩn xác
                batch_items = [dataset[idx] for idx in range(batch_start, batch_end)]
                
                # 1. XỬ LÝ AI HÀNG LOẠT (BATCH INFERENCE) - CHÌA KHÓA TĂNG TỐC TẠI ĐÂY
                # Gom toàn bộ ảnh trong mẻ này thành 1 list
                images = [item['image'].convert("RGB") for item in batch_items]
                
                # Tiền xử lý và đẩy 64 ảnh vào mô hình cùng 1 TÍCH TẮC
                inputs = processor(text=None, images=images, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                
                # --- THÊM DÒNG NÀY ĐỂ BÓC TÁCH TENSOR RA KHỎI OBJECT ---
                embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs

                # Lấy ra danh sách 64 vector
                batch_vectors = embeddings.cpu().numpy().tolist()
                
                # 2. ĐÓNG GÓI METADATA VÀ GỬI LÊN PINECONE
                vectors_to_upsert = []
                for j, item in enumerate(batch_items):
                    real_id = start_offset + batch_start + j
                    vec = batch_vectors[j]
                    
                    if namespace == "fashion_ns":
                        title = item.get('productDisplayName', f"Fashion Item {real_id}")
                        category = item.get('articleType', 'Clothing')
                    else:
                        title = item.get('ingredient', f"Food Item {real_id}")
                        category = item.get('category', 'Food')
                    
                    meta = {
                        "title": title,
                        "category": category,
                        "price": float(random.randint(10, 200)), 
                        "city": random.choice([default_city, "DaNang"]), 
                        "created_at": int((datetime.now() - timedelta(days=random.randint(0, 30))).timestamp()) 
                    }
                    
                    vectors_to_upsert.append((f"{prefix}-{real_id}", vec, meta))
                    st.session_state.local_db[namespace].append({"id": f"{prefix}-{real_id}", "vec": vec})
                
                # Pinecone sẽ nhận 1 cục 64 vectors siêu mượt
                index.upsert(vectors=vectors_to_upsert, namespace=namespace)

        with st.spinner("Đang nạp Fashion Data (Tenant A)..."):
            process_and_upsert(ds_fashion, "fashion_ns", "fsh", "Hanoi")
        with st.spinner("Đang nạp Food Data (Tenant B)..."):
            process_and_upsert(ds_food, "food_ns", "foo", "HCM")
            
        speed = (data_size * 2) / (time.time() - start_upsert)
        st.success(f"🚀 Indexing Speed: {speed:.2f} vectors/giây")
        
        # Cộng dồn tổng dữ liệu
        st.session_state.total_data += (data_size * 2)
        
        # LƯU VÀO HISTORY CÓ CỘT "HÀNH ĐỘNG"
        st.session_state.history.append({
            "Hành động": "🚀 Cold Start" if btn_reset else "➕ Nạp thêm",
            "Data Size (Total)": st.session_state.total_data,
            "Provisioning (s)": round(prov_time, 2),
            "Speed (vec/s)": round(speed, 2)
        })
        
        st.rerun()

# --- KHOẢNG TRỐNG HIỂN THỊ LỊCH SỬ SCALING ---
st.divider()
st.subheader("📈 Scaling History & Data Volume Growth")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    
    col_tbl, col_chart = st.columns([1, 2])
    with col_tbl:
        st.dataframe(df_history, use_container_width=True)
    with col_chart:
        # Vẽ biểu đồ Line Chart
        chart_data = df_history.set_index("Data Size (Total)")["Speed (vec/s)"]
        st.line_chart(chart_data)
else:
    st.info("💡 Hãy nạp dữ liệu bằng nút 'Reset' sau đó dùng 'Nạp thêm' để theo dõi biểu đồ Scaling liên tục!")

# --- 4. MAIN SEARCH & RESULTS ---
st.subheader("🔍 Search & Performance Dashboard")

c_filt1, c_filt2, c_filt3, c_filt4, c_filt5 = st.columns(5)
with c_filt1: target_ns = st.selectbox("Namespace (Multi-tenancy):", ["fashion_ns", "food_ns"])
with c_filt2: city_filter = st.multiselect("Geography:", ["Hanoi", "HCM", "DaNang"], default=["Hanoi", "HCM", "DaNang"])
with c_filt3: max_price = st.slider("Max Price ($):", 10, 200, 200)
with c_filt4: is_recent = st.checkbox("Chỉ hàng mới (7 ngày qua)")
with c_filt5: top_k = st.number_input("Số lượng kết quả:", min_value=1, max_value=20, value=8)

query = st.text_input("Query:", "green")

if st.button("Run Search & Benchmark", type="primary"):
    if not api_key: st.error("Vui lòng nhập API Key ở thanh bên!"); st.stop()
    index = Pinecone(api_key=api_key).Index(index_name)
    
    q_vec = get_text_embedding(query)
    
    p_filter = {
        "city": {"$in": city_filter},
        "price": {"$lte": max_price}
    }
    if is_recent:
        p_filter["created_at"] = {"$gte": int((datetime.now() - timedelta(days=7)).timestamp())}
    
    # 1. TRUY VẤN THỰC TẾ (RẤT NHANH)
    t1 = time.time()
    res = index.query(vector=q_vec, top_k=top_k, namespace=target_ns, include_metadata=True, filter=p_filter)
    latency = (time.time() - t1) * 1000
    
    recall_rate = calculate_recall(res, q_vec, target_ns, top_k=top_k)
    
    # --- TẠO PLACEHOLDER ĐỂ SẮP XẾP LẠI GIAO DIỆN ---
    metrics_placeholder = st.container() # Giữ chỗ cho đồng hồ đo ở trên
    results_placeholder = st.container() # Giữ chỗ cho kết quả ảnh ở dưới

    # 2. VẼ KẾT QUẢ TÌM KIẾM NGAY LẬP TỨC VÀO PLACEHOLDER BÊN DƯỚI
    with results_placeholder:
        st.divider()
        st.subheader(f"🛍️ {len(res['matches'])} kết quả (Tải xong trong {latency:.0f} ms!)")
        
        if res['matches']:
            n_cols = 4
            for i in range(0, len(res['matches']), n_cols):
                batch = res['matches'][i : i + n_cols]
                cols = st.columns(n_cols) 
                
                for j, m in enumerate(batch):
                    with cols[j]:
                        with st.container(border=True):
                            idx = int(m['id'].split('-')[1])
                            is_fashion = "fsh" in m['id']
                            current_dataset = ds_fashion if is_fashion else ds_food
                            
                            # Thuật toán an toàn: Xoay vòng ảnh nếu ID nạp thêm vượt quá kích thước Dataset gốc
                            safe_idx = idx % len(current_dataset)
                            img = current_dataset[safe_idx]['image']
                            st.image(img, use_container_width=True)
                            
                            meta = m['metadata']
                            st.markdown(f"**{meta.get('title', 'N/A')}**")
                            st.caption(f"🏷️ {meta.get('category', 'N/A')}")
                            st.markdown(f"💰 **<span style='color:green'>${meta.get('price', 0)}</span>** | 📍 {meta.get('city', 'N/A')}", unsafe_allow_html=True)
                            
                            match_percent = min(m['score'], 1.0)
                            st.progress(match_percent, text=f"🎯 Score: {match_percent*100:.1f}%")
        else:
            st.warning("Không tìm thấy kết quả phù hợp với bộ lọc!")

    # 3. SAU KHI VẼ ẢNH XONG, MỚI QUAY LẠI CHỖ TRỐNG Ở TRÊN ĐỂ CHẠY BENCHMARK
    with metrics_placeholder:
        st.divider()
        with st.spinner("⏳ Khách hàng đã nhận được kết quả. Đang chạy Benchmark ngầm (QPS & Stress Test) để đo hiệu năng..."):
            avg_latency = benchmark_avg(index, q_vec, target_ns)
            qps = measure_qps(index, q_vec, target_ns)
            
            stress_latency = None
            if run_stress:
                t_start = time.time()
                for _ in range(num_requests):
                    index.query(vector=q_vec, top_k=top_k, namespace=target_ns)
                stress_latency = ((time.time() - t_start) / num_requests) * 1000

        # VẼ 5 CÁI ĐỒNG HỒ ĐO SAU KHI BENCHMARK XONG
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("⚡ Current Latency", f"{latency:.2f} ms")
        m2.metric("📊 Avg Latency", f"{avg_latency:.2f} ms")
        m3.metric("🚀 QPS", f"{qps:.2f}")
        
        recall_color = "normal" if recall_rate >= 80 else "off"
        m4.metric("🎯 Recall Rate", f"{recall_rate:.1f} %", delta="ANN vs k-NN", delta_color=recall_color)
        
        if stress_latency:
            m5.metric("🔥 Stress Latency", f"{stress_latency:.2f} ms", delta=f"{stress_latency - avg_latency:.1f} ms", delta_color="inverse")