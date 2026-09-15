from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from connectome_lab.data import PROCESSED, load_graph
from connectome_lab.simulation import ConnectomeReservoir, simulate_lif


st.set_page_config(page_title="Fly Connectome Lab", page_icon="🪰", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 15% 0%, #162822 0, #0b1110 38%, #080b0b 100%); }
    [data-testid="stMetric"] { background:#111b18; border:1px solid #24483d; padding:14px; border-radius:12px; }
    h1,h2,h3 { letter-spacing:-.025em; }
    .small-note { color:#9db3ab; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def graph_data():
    return load_graph()


def network_figure(graph, count: int) -> go.Figure:
    keep = (graph.source < count) & (graph.target < count)
    src, dst, weights = graph.source[keep], graph.target[keep], graph.weight[keep]
    if weights.size > 650:
        take = np.argpartition(weights, -650)[-650:]
        src, dst, weights = src[take], dst[take], weights[take]
    net = nx.DiGraph()
    net.add_nodes_from(range(count))
    net.add_weighted_edges_from(zip(src.tolist(), dst.tolist(), weights.tolist()))
    pos = nx.spring_layout(net, seed=8, iterations=45, weight="weight")
    edge_x, edge_y = [], []
    for a, b in net.edges:
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    node_x = [pos[i][0] for i in range(count)]
    node_y = [pos[i][1] for i in range(count)]
    degree = np.array([net.degree(i) for i in range(count)])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#315248", width=.6), hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=5 + np.sqrt(degree) * 2, color=degree, colorscale="Viridis", line_width=0),
            text=[f"{graph.labels[i]}<br>body {graph.body_ids[i]}<br>degree {degree[i]}" for i in range(count)],
            hoverinfo="text",
        )
    )
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_visible=False, yaxis_visible=False)
    return fig


try:
    graph = graph_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code(".\\setup.ps1")
    st.stop()

st.title("🪰 Fly Connectome Lab")
st.caption("실제 MaleCNS v1.0 연결망을 탐색하고, 단순 신경 모델과 게임 제어기로 체험하는 로컬 실험실")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("로컬 뉴런", f"{graph.neuron_count:,}")
c2.metric("가중 연결", f"{graph.edge_count:,}")
c3.metric("공식 원본", "MaleCNS v1.0")
c4.metric("실행 환경", "CPU 홈 서버" if os.environ.get("DEPLOYMENT_MODE") == "server" else "RTX / CUDA")
c5.metric("라이선스", "CC BY")

with st.sidebar:
    st.header("중요한 구분")
    st.info("실제 데이터는 **연결 구조와 시냅스 수**입니다. 전압 방정식, 감각 입력, 운동 출력과 학습 규칙은 이 앱이 만든 실험 모델입니다.")
    st.caption(f"처리 데이터: {PROCESSED}")
    st.link_button("MaleCNS 공식 설명", "https://male-cns.janelia.org/")
    st.link_button("공식 다운로드", "https://male-cns.janelia.org/download/")

overview, explore, simulation, learning = st.tabs(["시작 · 3D", "연결망 탐색", "신경 시뮬레이션", "학습 실험"])

with overview:
    st.subheader("세 가지 방식으로 바로 체험하세요")
    left, mid, right = st.columns(3)
    with left:
        st.markdown("### 1. 실제 3D 뇌")
        st.write("설치 없이 공식 Neuroglancer에서 EM 영상과 뉴런 분할을 봅니다.")
        st.link_button("MaleCNS 3D 열기 ↗", "https://male-cns.janelia.org/explore/", use_container_width=True)
    with mid:
        st.markdown("### 2. 로컬 신경망")
        st.write("아래 탭에서 실제 연결의 허브, 세포 이름과 신호 전파를 확인합니다.")
        st.success("1.1GB 전체 연결표에서 상위 회로를 로컬 추출했습니다.")
    with right:
        st.markdown("### 3. Connectome Saber")
        st.write("실제 연결 구조를 reservoir로 쓰는 데스크톱 동반 리듬게임입니다.")
        st.code(".\\run-rhythm.ps1", language="powershell")
    st.divider()
    st.markdown("""
    **외부 탐색기**

    - [neuPrint: 질의와 경로 분석](https://neuprint.janelia.org/?dataset=male-cns:v1.0)
    - [Virtual Fly Brain: 데이터셋 통합 탐색](https://v2.virtualflybrain.org/)
    - [Codex: 여러 초파리 커넥톰 비교](https://codex.flywire.ai/)
    """)

with explore:
    st.subheader("강하게 연결된 뉴런들의 로컬 네트워크")
    count = st.slider("표시할 뉴런 수", 30, 220, 90, 10)
    st.plotly_chart(network_figure(graph, count), use_container_width=True)
    rows = pd.DataFrame({"body_id": graph.body_ids[:count], "cell_label": graph.labels[:count]})
    st.dataframe(rows, use_container_width=True, hide_index=True)

with simulation:
    st.subheader("단순 LIF형 신호 전파")
    st.caption("생물학적 재현이 아니라 실제 배선을 사용하는 교육용 동역학입니다.")
    ca, cb, cc = st.columns(3)
    neurons = ca.slider("뉴런", 200, min(2500, graph.neuron_count), 900, 100)
    steps = cb.slider("시간 스텝", 80, 500, 240, 20)
    strength = cc.slider("입력 세기", .5, 2.5, 1.4, .1)
    if st.button("자극 시작", type="primary"):
        with st.spinner("희소 신경망 계산 중..."):
            spikes, voltage = simulate_lif(graph.adjacency(neurons), steps=steps, stimulus=strength)
        active = np.flatnonzero(spikes.sum(axis=0))[:140]
        raster = spikes[:, active].T if active.size else np.zeros((1, steps))
        fig = go.Figure(go.Heatmap(z=raster, colorscale=[[0, "#08100d"], [1, "#71f6bd"]], showscale=False))
        fig.update_layout(height=430, xaxis_title="time step", yaxis_title="active neuron", margin=dict(l=30, r=10, t=10, b=35))
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"총 스파이크 **{int(spikes.sum()):,}**, 활성 뉴런 **{int((spikes.sum(axis=0)>0).sum()):,}**")

with learning:
    st.subheader("커넥톰을 AI 골격으로 사용하기")
    st.write("실제 연결망을 고정 reservoir로 두고, 네 개 감각 신호를 네 운동 방향으로 바꾸는 선형 readout만 학습합니다.")
    st.success("CUDA PyTorch가 설치되어 있습니다. 전체 로컬 회로 GPU 테스트: `.\\.venv\\Scripts\\python.exe -m scripts.gpu_demo`")
    nlearn = st.slider("학습에 쓸 뉴런", 120, min(700, graph.neuron_count), 320, 40)
    if st.button("운동 readout 학습", type="primary"):
        with st.spinner("connectome reservoir에서 특징을 수집하고 있습니다..."):
            model = ConnectomeReservoir(graph.adjacency(nlearn))
            error = model.train_motor_readout(samples=700)
            outputs = []
            for lane in range(4):
                model.reset()
                obs = np.zeros(4, dtype=np.float32); obs[lane] = 1
                for _ in range(8): out = model.step(obs)
                outputs.append(out)
        expected = np.array([[-1,1],[1,1],[-1,-1],[1,-1]])
        result = pd.DataFrame({"input_lane": range(4), "target_x": expected[:,0], "target_y": expected[:,1], "output_x": np.array(outputs)[:,0], "output_y": np.array(outputs)[:,1]})
        st.metric("학습 샘플 평균 오차", f"{error:.4f}")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.success("같은 원리를 리듬게임에서 실시간으로 사용합니다. PowerShell에서 `.\\run-rhythm.ps1`를 실행하세요.")
