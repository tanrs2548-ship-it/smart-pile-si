import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Smart Pile & Load Transfer Simulator",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ AI-Assisted Boring Log to Pile Design & Load Transfer Simulator")
st.markdown("ระบบวิเคราะห์กำลังรับน้ำหนักเสาเข็ม จำลองพฤติกรรมการถ่ายแรง ($t-z / q-z$) และประเมินความเสี่ยงเชิงความน่าจะเป็น (Reliability Analysis)")

# ==========================================
# 2. GEOTECHNICAL & LOAD TRANSFER ENGINE
# ==========================================
def get_soil_parameters(layer):
    """แปลงค่า SPT-N และชนิดดินเป็นพารามิเตอร์ทางวิศวกรรมปฐพี (Empirical Correlations)"""
    soil_type = layer['Soil Type']
    N = float(layer['SPT_N'])
    gamma = float(layer['Unit Weight (kN/m3)'])
    
    if soil_type == 'Clay':
        # Undrained Shear Strength Su (kPa) ~ 6.0 * N (Terzaghi & Peck / local correlation)
        Su = max(10.0, 6.0 * N)
        # Adhesion factor alpha (API RP 2GEO / Tomlinson)
        alpha = 1.0 if Su <= 25 else (0.5 if Su >= 75 else 1.0 - 0.5 * ((Su - 25) / 50))
        fs_ult = alpha * Su
        qb_ult = 9.0 * Su
        return {'Su': Su, 'phi': 0.0, 'fs_ult': fs_ult, 'qb_ult': qb_ult, 'gamma': gamma}
    else:  # Sand / Silt
        # Effective friction angle phi (deg)
        phi = min(45.0, 27.1 + 0.3 * N - 0.00054 * (N ** 2))
        phi_rad = np.radians(phi)
        # Earth pressure coefficient Ks & delta (Bored Pile)
        Ks = 0.7
        delta = 0.75 * phi_rad
        # Bearing capacity factor Nq (Berezentzev)
        Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.radians(45 + phi / 2)) ** 2)
        return {'Su': 0.0, 'phi': phi, 'Ks': Ks, 'delta': delta, 'Nq': Nq, 'gamma': gamma}

def tz_curve_api(z, z_peak, t_max, soil_type):
    """สมการความสัมพันธ์แรงเสียดทานกับระยะเคลื่อนตัว (API RP 2GEO t-z curve)"""
    if z <= 0:
        return 0.0
    ratio = z / z_peak
    if soil_type == 'Clay':
        # Clay exhibits post-peak softening
        if ratio <= 0.16:
            return 0.30 * t_max * (ratio / 0.16)
        elif ratio <= 0.31:
            return (0.30 + 0.20 * ((ratio - 0.16) / 0.15)) * t_max
        elif ratio <= 0.57:
            return (0.50 + 0.25 * ((ratio - 0.31) / 0.26)) * t_max
        elif ratio <= 0.80:
            return (0.75 + 0.15 * ((ratio - 0.57) / 0.23)) * t_max
        elif ratio <= 1.00:
            return (0.90 + 0.10 * ((ratio - 0.80) / 0.20)) * t_max
        elif ratio <= 2.00:
            return (1.00 - 0.20 * ((ratio - 1.00) / 1.00)) * t_max  # Residual 0.80 t_max
        else:
            return 0.80 * t_max
    else:
        # Sand: Elastic-plastic hyperbolic approximation
        if ratio < 1.0:
            return t_max * np.sin((np.pi / 2.0) * ratio)
        else:
            return t_max

def qz_curve_api(z_tip, D, q_max):
    """สมการความสัมพันธ์แรงต้านทานปลายเสาเข็มกับระยะทรุดตัวที่ปลาย (API q-z curve)"""
    if z_tip <= 0:
        return 0.0
    z_ult = 0.10 * D  # Ultimate tip resistance mobilized at 10% of diameter
    ratio = z_tip / z_ult
    if ratio <= 0.02:
        return 0.25 * q_max * (ratio / 0.02)
    elif ratio <= 0.13:
        return (0.25 + 0.25 * ((ratio - 0.02) / 0.11)) * q_max
    elif ratio <= 0.42:
        return (0.50 + 0.25 * ((ratio - 0.13) / 0.29)) * q_max
    elif ratio <= 1.00:
        return (0.75 + 0.25 * ((ratio - 0.42) / 0.58)) * q_max
    else:
        return q_max

def run_load_transfer_simulation(df_soil, D, L, Ep_GPa, gwt_depth, n_elements=40):
    """จำลอง Load-Settlement Curve และ Axial Load Profile ด้วยวิธี Finite Difference (t-z/q-z)"""
    dz = L / n_elements
    area = np.pi * (D ** 2) / 4.0
    perimeter = np.pi * D
    Ep = Ep_GPa * 1e6  # Convert GPa to kPa (kN/m2)
    
    # เตรียมข้อมูลชั้นดินของแต่ละ Element
    elem_depths = np.linspace(dz / 2.0, L - dz / 2.0, n_elements)
    elem_soil_types = []
    elem_tmax = []
    sigma_v_eff = 0.0
    
    # คำนวณ Effective Stress และ Ultimate Unit Skin Friction ตามความลึก
    for d_mid in elem_depths:
        layer_match = df_soil[(df_soil['Top (m)'] <= d_mid) & (df_soil['Bottom (m)'] >= d_mid)]
        if layer_match.empty:
            layer = df_soil.iloc[-1]
        else:
            layer = layer_match.iloc[0]
            
        params = get_soil_parameters(layer)
        gamma_eff = params['gamma'] - 9.81 if d_mid > gwt_depth else params['gamma']
        sigma_v_eff += gamma_eff * dz
        
        if layer['Soil Type'] == 'Clay':
            fs = params['fs_ult']
        else:
            fs = min(120.0, params['Ks'] * sigma_v_eff * np.tan(params['delta']))  # Limit skin friction at 120 kPa
            
        elem_soil_types.append(layer['Soil Type'])
        elem_tmax.append(fs * perimeter * dz)  # แรงเสียดทานสูงสุดของ Element (kN)
        
    # หาค่า Ultimate End Bearing ที่ปลายเสาเข็ม (ความลึก L)
    tip_layer_match = df_soil[(df_soil['Top (m)'] <= L) & (df_soil['Bottom (m)'] >= L)]
    tip_layer = df_soil.iloc[-1] if tip_layer_match.empty else tip_layer_match.iloc[0]
    tip_params = get_soil_parameters(tip_layer)
    
    if tip_layer['Soil Type'] == 'Clay':
        qb_ult = tip_params['qb_ult']
    else:
        qb_ult = min(10000.0, tip_params['Nq'] * sigma_v_eff)  # Limit end bearing at 10,000 kPa
    Qb_max = qb_ult * area
    Qs_max = sum(elem_tmax)
    Q_ult_total = Qs_max + Qb_max

    # สร้าง Load-Settlement Curve โดยไล่กำหนดระยะเคลื่อนตัวที่ปลายเสาเข็ม (z_tip) จาก 0 ถึง 0.12*D
    z_tips = np.linspace(0.0001, 0.12 * D, 50)
    top_loads = []
    top_settlements = []
    profiles = []  # เก็บโปรไฟล์แรงตามความลึกในแต่ละขั้น
    
    z_peak = 0.01 * D  # ระยะเคลื่อนตัวที่เกิด skin friction สูงสุด (~1% D)
    
    for z_t in z_tips:
        Q_bot = qz_curve_api(z_t, D, Qb_max)
        z_curr = z_t
        
        depth_nodes = [L]
        load_nodes = [Q_bot]
        
        for i in range(n_elements - 1, -1, -1):
            z_mid = z_curr
            for _ in range(5):
                T_side = tz_curve_api(z_mid, z_peak, elem_tmax[i], elem_soil_types[i])
                Q_top = Q_bot + T_side
                Q_avg = 0.5 * (Q_bot + Q_top)
                delta_z = (Q_avg * dz) / (area * Ep)
                z_mid = z_curr + 0.5 * delta_z
                
            z_curr += delta_z
            Q_bot = Q_top
            depth_nodes.append(elem_depths[i] - dz / 2.0)
            load_nodes.append(Q_top)
            
        top_loads.append(Q_bot)
        top_settlements.append(z_curr * 1000.0)  # แปลงเป็น mm
        profiles.append((depth_nodes[::-1], load_nodes[::-1]))
        
    return {
        'Q_ult': Q_ult_total,
        'Qs_max': Qs_max,
        'Qb_max': Qb_max,
        'top_loads': np.array(top_loads),
        'top_settlements': np.array(top_settlements),
        'profiles': profiles
    }

def run_monte_carlo_reliability(Qs_mean, Qb_mean, P_design, cov_soil=0.20, cov_load=0.10, n_sim=2000):
    """จำลอง Monte Carlo 2,000 รอบ เพื่อคำนวณ Probability of Failure (Pf) และ Reliability Index (Beta)"""
    np.random.seed(42)
    R_mean = Qs_mean + Qb_mean
    R_std = R_mean * cov_soil
    sigma_ln_R = np.sqrt(np.log(1 + (R_std / R_mean) ** 2))
    mu_ln_R = np.log(R_mean) - 0.5 * (sigma_ln_R ** 2)
    R_samples = np.random.lognormal(mu_ln_R, sigma_ln_R, n_sim)
    
    Q_std = P_design * cov_load
    Q_samples = np.random.normal(P_design, Q_std, n_sim)
    
    g = R_samples - Q_samples
    failures = np.sum(g <= 0)
    Pf = (failures / n_sim) * 100.0
    beta = (np.mean(g)) / np.std(g)
    return R_samples, Q_samples, Pf, beta

# ==========================================
# 3. SIDEBAR INPUTS (USER INTERFACE)
# ==========================================
with st.sidebar:
    st.header("1. ข้อมูลเสาเข็มและน้ำหนักบรรทุก")
    pile_type = st.selectbox("ประเภทเสาเข็ม", ["Bored Pile (เสาเข็มเจาะ)", "Driven Spun Pile (เสาเข็มตอก)"])
    D = st.number_input("เส้นผ่านศูนย์กลางเสาเข็ม D (m)", min_value=0.20, max_value=2.00, value=0.60, step=0.05)
    L = st.number_input("ความยาวเสาเข็ม L (m)", min_value=3.0, max_value=50.0, value=16.0, step=0.5)
    Ep_GPa = st.number_input("โมดูลัสยืดหยุ่นคอนกรีต Ep (GPa)", min_value=15.0, max_value=45.0, value=25.0, step=1.0)
    gwt_depth = st.number_input("ระดับน้ำใต้ดิน GWT (m)", min_value=0.0, max_value=30.0, value=2.5, step=0.5)
    
    st.divider()
    st.header("2. แรงกระทำและค่าความแปรปรวน")
    P_design = st.number_input("น้ำหนักบรรทุกใช้งานต่อต้น P (kN)", min_value=50.0, max_value=10000.0, value=950.0, step=50.0)
    FS_req = st.number_input("Factor of Safety ที่ต้องการ (FS)", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
    cov_soil = st.slider("ความแปรปรวนของชั้นดิน (CoV Soil)", 0.05, 0.40, 0.20, 0.05)

# ==========================================
# 4. MAIN AREA: BORING LOG EDITOR
# ==========================================
st.subheader("📋 ตารางข้อมูลชั้นดิน (Boring Log Data)")
col_preset, _ = st.columns([1, 2])
with col_preset:
    preset = st.selectbox("เลือกโปรไฟล์ชั้นดินตัวอย่าง (หรือแก้ไขในตารางด้านล่างได้ทันที)", [
        "โปรไฟล์ A: ดินเหนียวปนทรายโคราช (Nakhon Ratchasima Profile)",
        "โปรไฟล์ B: ดินเหนียวอ่อนกรุงเทพฯ (Bangkok Soft Clay Profile)"
    ])

if "โปรไฟล์ A" in preset:
    default_data = pd.DataFrame({
        'Top (m)': [0.0, 3.0, 7.0, 12.0, 18.0],
        'Bottom (m)': [3.0, 7.0, 12.0, 18.0, 25.0],
        'Soil Type': ['Clay', 'Sand', 'Clay', 'Sand', 'Sand'],
        'SPT_N': [6, 14, 18, 28, 42],
        'Unit Weight (kN/m3)': [17.5, 18.5, 19.0, 19.8, 20.5]
    })
else:
    default_data = pd.DataFrame({
        'Top (m)': [0.0, 2.0, 11.0, 16.0, 22.0],
        'Bottom (m)': [2.0, 11.0, 16.0, 22.0, 30.0],
        'Soil Type': ['Clay', 'Clay', 'Clay', 'Sand', 'Sand'],
        'SPT_N': [4, 2, 12, 25, 38],
        'Unit Weight (kN/m3)': [16.5, 15.8, 18.2, 19.5, 20.2]
    })

df_soil = st.data_editor(
    default_data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Soil Type": st.column_config.SelectboxColumn("Soil Type", options=["Clay", "Sand"], required=True),
        "SPT_N": st.column_config.NumberColumn("SPT-N (blows/ft)", min_value=0, max_value=100, required=True)
    }
)

# ==========================================
# 5. RUN SIMULATION & DISPLAY DASHBOARD
# ==========================================
sim_results = run_load_transfer_simulation(df_soil, D, L, Ep_GPa, gwt_depth)
Q_ult = sim_results['Q_ult']
Q_allow = Q_ult / FS_req
FS_actual = Q_ult / P_design

settlement_at_P = np.interp(P_design, sim_results['top_loads'], sim_results['top_settlements'])

R_samples, Q_samples, Pf, beta = run_monte_carlo_reliability(
    sim_results['Qs_max'], sim_results['Qb_max'], P_design, cov_soil=cov_soil
)

st.divider()
st.subheader("📊 สรุปผลการวิเคราะห์เสาเข็มและความปลอดภัย (Design Summary)")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("กำลังประลัยรวม (Q_ult)", f"{Q_ult:,.1f} kN", f"Skin: {sim_results['Qs_max']:,.0f} | Tip: {sim_results['Qb_max']:,.0f}")
m2.metric("กำลังรับน้ำหนักปลอดภัย (Q_allow)", f"{Q_allow:,.1f} kN", f"FS = {FS_req:.1f}")
m3.metric("FS จริงของระบบ", f"{FS_actual:.2f}", "ผ่านเกณฑ์" if FS_actual >= FS_req else "ไม่ผ่านเกณฑ์!", delta_color="normal" if FS_actual >= FS_req else "inverse")
m4.metric("การทรุดตัวที่น้ำหนักใช้งาน", f"{settlement_at_P:.2f} mm", "ปกติ (< 25 mm)" if settlement_at_P < 25 else "ทรุดตัวสูง!", delta_color="normal" if settlement_at_P < 25 else "inverse")
m5.metric("โอกาสวิบัติ (Pf / Reliability β)", f"{Pf:.2f}%", f"β = {beta:.2f}", delta_color="normal" if Pf < 1.0 else "inverse")

# ==========================================
# 6. INTERACTIVE PLOTS (PLOTLY)
# ==========================================
st.subheader("📈 กราฟจำลองพฤติกรรมการถ่ายแรงและการทรุดตัว (Interactive Simulator)")

load_step_idx = st.slider(
    "เลื่อนแถบเพื่อจำลองการเพิ่มน้ำหนักบรรทุกที่หัวเสาเข็ม (Step Loading Simulation):",
    min_value=1,
    max_value=len(sim_results['top_loads']) - 1,
    value=15
)

selected_load = sim_results['top_loads'][load_step_idx]
selected_settle = sim_results['top_settlements'][load_step_idx]
depths_prof, loads_prof = sim_results['profiles'][load_step_idx]

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=sim_results['top_loads'],
        y=sim_results['top_settlements'],
        mode='lines',
        name='Load-Settlement Curve',
        line=dict(color='#1f77b4', width=3)
    ))
    fig1.add_trace(go.Scatter(
        x=[selected_load],
        y=[selected_settle],
        mode='markers+text',
        name='Current Load Step',
        text=[f"  {selected_load:,.0f} kN ({selected_settle:.1f} mm)"],
        textposition="middle right",
        marker=dict(color='red', size=12)
    ))
    fig1.add_vline(x=P_design, line_dash="dash", line_color="green", annotation_text=f"Design Load ({P_design:.0f} kN)")
    fig1.update_layout(
        title="1. กราฟน้ำหนักบรรทุก vs การทรุดตัวที่หัวเสาเข็ม (Load-Settlement)",
        xaxis_title="Axial Load at Pile Head (kN)",
        yaxis_title="Pile Head Settlement (mm)",
        yaxis=dict(autorange="reversed"),
        height=450
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_g2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=loads_prof,
        y=depths_prof,
        mode='lines+markers',
        name='Axial Force in Pile',
        fill='tozerox',
        line=dict(color='#ff7f0e', width=3)
    ))
    fig2.update_layout(
        title=f"2. กราฟการถ่ายแรงตามความลึก (ที่น้ำหนักหัวเสาเข็ม {selected_load:,.0f} kN)",
        xaxis_title="Axial Load Remaining in Pile (kN)",
        yaxis_title="Depth from Ground Surface (m)",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, max(sim_results['top_loads']) * 1.1]),
        height=450
    )
    st.plotly_chart(fig2, use_container_width=True)

fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=R_samples, name='Pile Capacity Distribution (R)', opacity=0.65, marker_color='#2ca02c', nbinsx=40))
fig3.add_trace(go.Histogram(x=Q_samples, name='Applied Load Distribution (Q)', opacity=0.65, marker_color='#d62728', nbinsx=40))
fig3.update_layout(
    barmode='overlay',
    title=f"3. การวิเคราะห์ความเสี่ยงเชิงความน่าจะเป็น (Monte Carlo 2,000 รอบ | โอกาสวิบัติ Pf = {Pf:.2f}%, β = {beta:.2f})",
    xaxis_title="Force (kN)",
    yaxis_title="Frequency",
    height=350
)
st.plotly_chart(fig3, use_container_width=True)
