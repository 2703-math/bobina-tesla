"""
⚡ Transformadores, Bobina de Tesla e Transmissão de Energia
============================================================
Executar com:  streamlit run transformador_tesla.py

Três módulos que se completam (todos com física calculada, não desenhada à mão):
  1. Transformador  – relação de espiras, fluxo no núcleo, potência conservada, e por que NÃO funciona em CC
  2. Bobina de Tesla – dois circuitos LC acoplados (RK4): sintonia, transferência de energia e faíscas
  3. Transmissão de energia – por que a rede usa alta tensão (perdas I²R)

Roteiro pedagógico sugerido: Prever → Observar → Explicar (use as caixas "Teste sua intuição").
"""
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================
# PÁGINA E ESTILO
# ============================================
st.set_page_config(page_title="Física Visual: Transformador, Tesla e Rede Elétrica", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; text-align: center; margin-bottom: 0.3rem; }
    .subtitle   { font-size: 1.1rem; color: #64748b; text-align: center; margin-bottom: 1.5rem; }
    .concept-card { background: #f8fafc; border-radius: 12px; padding: 1.2rem; border-left: 4px solid #3b82f6;
                    margin-bottom: 1rem; color: #334155; }
    .alert-card   { background: #fffbeb; border-radius: 12px; padding: 1.2rem; border-left: 4px solid #f59e0b;
                    margin-bottom: 1rem; color: #334155; }
    .highlight { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

COR_P, COR_S, COR_FLUXO = "#f97316", "#10b981", "#2563eb"   # primário, secundário, fluxo
N_QUADROS = 60


# ============================================
# FÍSICA — TRANSFORMADOR
# ============================================
FREQ_REDE = 60.0                 # Hz
V_LAMP, P_LAMP = 127.0, 60.0     # lâmpada de 127 V / 60 W
R_LAMP = V_LAMP**2 / P_LAMP      # ≈ 269 Ω
TAU_CC = 4e-3                    # constante de tempo do primário em CC (s)


def sinais_transformador(fonte, Np, Ns, Vp_rms, n):
    """Tensões (V) e fluxo (mWb) em ~2 ciclos de 60 Hz. Lei de Faraday: v = N·dΦ/dt."""
    T_ms = 1000.0 / FREQ_REDE
    t_ms = np.linspace(0, 2 * T_ms, n)
    ts = t_ms / 1000.0
    w = 2 * np.pi * FREQ_REDE
    if fonte == "CA":
        vp_pk = Vp_rms * np.sqrt(2)
        vp = vp_pk * np.sin(w * ts)
        vs = vp * Ns / Np
        phi = -vp_pk / (Np * w) * np.cos(w * ts)                   # Φ = ∫v dt / N
    else:  # bateria ligada em t = 0: o fluxo cresce e satura; só há tensão induzida no "liga"
        vp = np.full(n, float(Vp_rms))
        vs = Vp_rms * Ns / Np * np.exp(-ts / TAU_CC)
        phi = Vp_rms * TAU_CC / Np * (1 - np.exp(-ts / TAU_CC))
    return t_ms, vp, vs, phi * 1e3


def regime_transformador(Np, Ns, Vp, n_lamp, eta):
    Vs = Vp * Ns / Np
    R = R_LAMP / n_lamp                      # lâmpadas em paralelo
    Ps = Vs**2 / R
    Pin = Ps / eta
    return dict(Vs=Vs, Is=Vs / R, Ps=Ps, Pin=Pin, Ip=Pin / Vp, perdas=Pin - Ps)


# ============================================
# FÍSICA — BOBINA DE TESLA (dois circuitos LC acoplados)
# ============================================
L1, L2 = 25e-6, 50e-3                                   # indutâncias primária e secundária (H)
F2 = 150e3                                              # frequência de ressonância do secundário (Hz)
C2 = 1.0 / ((2 * np.pi * F2) ** 2 * L2)                 # capacitância do secundário + toroide (≈ 22,5 pF)
C1_SINTONIA = 1.0 / ((2 * np.pi * F2) ** 2 * L1)        # capacitor primário que sintoniza (≈ 45 nF)
R1_BASE, R2_BASE = 0.3, 400.0                           # resistências (Ω)
DT, T_SIM = 0.04e-6, 100e-6                             # passo e duração da simulação (s)


def simular_tesla(C1, V0, k, perdas):
    """RK4 para o sistema  [L1 M; M L2]·[i1'; i2'] = [-q1/C1 - R1·i1; -q2/C2 - R2·i2].
    C1 pode ser vetor (varre vários capacitores de uma vez). Retorna t e estados (n+1, 4, M)."""
    C1 = np.atleast_1d(np.asarray(C1, dtype=float))
    M = k * math.sqrt(L1 * L2)
    det = L1 * L2 - M * M
    R1, R2 = R1_BASE * perdas, R2_BASE * perdas

    def f(s):
        q1, i1, q2, i2 = s
        r1 = -q1 / C1 - R1 * i1
        r2 = -q2 / C2 - R2 * i2
        return np.array([i1, (L2 * r1 - M * r2) / det, i2, (L1 * r2 - M * r1) / det])

    n = int(round(T_SIM / DT))
    s = np.array([C1 * V0, np.zeros_like(C1), np.zeros_like(C1), np.zeros_like(C1)])   # capacitor primário carregado
    out = np.empty((n + 1, 4, C1.size))
    out[0] = s
    for j in range(n):
        k1 = f(s); k2 = f(s + 0.5 * DT * k1); k3 = f(s + 0.5 * DT * k2); k4 = f(s + DT * k3)
        s = s + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[j + 1] = s
    return np.arange(n + 1) * DT, out


@st.cache_data(show_spinner=False)
def dados_tesla(c1_nF, V0_kV, k, perdas):
    c1 = c1_nF * 1e-9
    t, out = simular_tesla(c1, V0_kV * 1e3, k, perdas)
    q1, i1, q2, i2 = out[:, 0, 0], out[:, 1, 0], out[:, 2, 0], out[:, 3, 0]
    V1, V2 = q1 / c1, q2 / C2
    w = int(round(1.0 / F2 / DT))                                   # janela ≈ 1 período da portadora

    def envelope(x):
        e = np.lib.stride_tricks.sliding_window_view(np.abs(x), w).max(axis=1)
        return np.pad(e, (w // 2, w - 1 - w // 2), mode="edge")

    def suave(x):
        um = np.ones(w) / w
        return np.convolve(x, um, mode="same") / np.convolve(np.ones_like(x), um, mode="same")

    Ep = suave(0.5 * q1**2 / c1 + 0.5 * L1 * i1**2)
    Es = suave(0.5 * q2**2 / C2 + 0.5 * L2 * i2**2)
    return dict(t_us=t * 1e6, V1=V1 / 1e3, V2=V2 / 1e3, env1=envelope(V1) / 1e3, env2=envelope(V2) / 1e3,
                fs=100 * Es / (Ep + Es), V2max=float(np.max(np.abs(V2)) / 1e3))


@st.cache_data(show_spinner=False)
def varredura_ressonancia(V0_kV, k, perdas):
    c1 = np.linspace(0.7, 1.3, 25) * C1_SINTONIA
    _, out = simular_tesla(c1, V0_kV * 1e3, k, perdas)
    return 1 / (2 * np.pi * np.sqrt(L1 * c1)) / 1e3, np.max(np.abs(out[:, 2, :] / C2), axis=0) / 1e3   # kHz, kV


# ============================================
# FÍSICA — TRANSMISSÃO DE ENERGIA
# ============================================
def calc_transmissao(P_MW, dist_km, V_kV, r_ohm_km):
    """Linha resistiva: P_env = P + R·I², I = P_env/V  ⇒  R·P_env²/V² − P_env + P = 0."""
    R = r_ohm_km * dist_km
    x = 4 * R * P_MW / V_kV**2
    base = dict(R=R, Vmin=math.sqrt(4 * R * P_MW))
    if x > 1:
        return dict(viavel=False, **base)
    Pe = V_kV**2 / (2 * R) * (1 - math.sqrt(1 - x))
    return dict(viavel=True, Pe=Pe, perda=Pe - P_MW, pct=100 * (Pe - P_MW) / Pe, I=Pe / V_kV * 1000, **base)


# ============================================
# COMPONENTES DE INTERFACE
# ============================================
def mostrar(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def botoes_animacao(duracao_ms, rotulo):
    return [{
        "type": "buttons", "showactive": False, "direction": "left",
        "x": 0.0, "y": 1.12, "xanchor": "left", "yanchor": "bottom",
        "buttons": [
            {"label": rotulo, "method": "animate",
             "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True,
                             "transition": {"duration": 0}, "mode": "immediate"}]},
            {"label": "❚❚ Pausar", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                               "transition": {"duration": 0}}]},
        ],
    }]


def slider_frames(rotulos, prefixo, sufixo):
    passos = [dict(method="animate", label=str(r),
                   args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                        transition=dict(duration=0))]) for i, r in enumerate(rotulos)]
    return [dict(active=0, x=0.0, len=1.0, y=-0.01, pad=dict(t=30, b=0), steps=passos,
                 currentvalue=dict(prefix=prefixo, suffix=sufixo, font=dict(size=13, color="#334155")),
                 font=dict(color="rgba(0,0,0,0)"), tickcolor="rgba(0,0,0,0)", ticklen=0, minorticklen=0)]


def pergunta(chave, enunciado, opcoes, correta, explicacao):
    """Mini-quiz: o aluno pode tentar de novo até acertar."""
    with st.expander("🎯 Teste sua intuição"):
        resp = st.radio(enunciado, opcoes, index=None, key=f"q_{chave}")
        if resp is not None:
            if opcoes.index(resp) == correta:
                st.success(f"✅ Isso mesmo! {explicacao}")
            else:
                st.warning("🤔 Ainda não. Volte à simulação, mexa nos controles, observe de novo e tente outra vez.")


class Construtor:
    """Agrupa a criação de traços estáticos e dinâmicos (evita erros de índice nos quadros da animação)."""

    def __init__(self, fig):
        self.fig, self.idx = fig, {}

    def add(self, nome, traco, row, col):
        self.fig.add_trace(traco, row=row, col=col)
        self.idx[nome] = len(self.fig.data) - 1

    def dinamicos(self, fixos, inicial):
        for nome, (cls, r, c, fixo) in fixos.items():
            self.add(nome, cls(**fixo, **inicial[nome]), r, c)

    def quadros(self, fixos, gerador, n):
        self.fig.frames = [go.Frame(name=str(i), traces=[self.idx[nm] for nm in fixos],
                                    data=[fixos[nm][0](**gerador(i)[nm]) for nm in fixos]) for i in range(n)]


def retangulo(x0, y0, x1, y1):
    return [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]


def circulo(cx, cy, r, n=40):
    a = np.linspace(0, 2 * np.pi, n)
    return list(cx + r * np.cos(a)), list(cy + r * np.sin(a))


# ============================================
# FIGURA 1 — TRANSFORMADOR
# ============================================
def perimetro(n, xa=4.6, ya=3.1):
    """n pontos igualmente espaçados sobre o circuito magnético do núcleo (sentido anti-horário)."""
    L, pts = 4 * xa + 4 * ya, []
    for i in range(n):
        s = i * L / n
        if s < 2 * xa:
            pts.append((-xa + s, -ya))
        elif s < 2 * xa + 2 * ya:
            pts.append((xa, -ya + (s - 2 * xa)))
        elif s < 4 * xa + 2 * ya:
            pts.append((xa - (s - 2 * xa - 2 * ya), ya))
        else:
            pts.append((-xa, ya - (s - 4 * xa - 2 * ya)))
    return pts


def espiras_xy(n_desenho, x0, x1):
    ys = [0.0] if n_desenho == 1 else list(np.linspace(-1.9, 1.9, n_desenho))
    xs, yy = [], []
    for y in ys:
        xs += [x0, x1, None]
        yy += [y, y, None]
    return xs, yy


def gerar_animacao_transformador(fonte, Np, Ns, Vp, n_lamp, eta, dur_ms):
    t_c, vp_c, vs_c, phi_c = sinais_transformador(fonte, Np, Ns, Vp, 300)
    t_f, vp_f, vs_f, phi_f = sinais_transformador(fonte, Np, Ns, Vp, N_QUADROS)
    reg = regime_transformador(Np, Ns, Vp, n_lamp, eta)
    phi_ref = max(float(np.max(np.abs(phi_c))), 1e-9)
    v_lim = 1.15 * max(float(np.max(np.abs(vp_c))), float(np.max(np.abs(vs_c))))
    phi_lim = 1.2 * phi_ref
    queimou = fonte == "CA" and reg["Vs"] > 1.3 * V_LAMP
    brilho = 0.0 if fonte == "CC" else min(1.0, (reg["Vs"] / V_LAMP) ** 2)
    vp_pk, vs_pk = Vp * math.sqrt(2), reg["Vs"] * math.sqrt(2)

    fig = make_subplots(rows=2, cols=2, column_widths=[0.6, 0.4], specs=[[{"rowspan": 2}, {}], [None, {}]],
                        horizontal_spacing=0.07, vertical_spacing=0.14,
                        subplot_titles=("Transformador (vista de frente)", "Tensões: primário × secundário",
                                        "Fluxo magnético no núcleo  Φ(t)"))
    b = Construtor(fig)

    # ---- fios ----
    xs_l = [8.6 + 1.2 * j for j in range(n_lamp)]
    fx, fy = [], []

    def fio(*pts):
        nonlocal fx, fy
        fx += [p[0] for p in pts] + [None]
        fy += [p[1] for p in pts] + [None]

    fio((-6.4, 1.9), (-7.2, 1.9), (-7.2, 3.4), (-9.5, 3.4), (-9.5, 0.9))
    fio((-6.4, -1.9), (-7.2, -1.9), (-7.2, -3.4), (-9.5, -3.4), (-9.5, -0.9))
    rail = xs_l[-1] + 0.6
    fio((6.4, 1.9), (7.0, 1.9), (7.0, 3.4), (rail, 3.4))
    fio((6.4, -1.9), (7.0, -1.9), (7.0, -3.4), (rail, -3.4))
    for x in xs_l:
        fio((x, 3.4), (x, 0.8))
        fio((x, -0.8), (x, -3.4))
    b.add("fios", go.Scatter(x=fx, y=fy, mode="lines", line=dict(color="#475569", width=3), hoverinfo="skip"), 1, 1)

    # ---- núcleo, bobinas ----
    ex, ey = retangulo(-6, -4, 6, 4)
    b.add("nucleo", go.Scatter(x=ex, y=ey, mode="lines", fill="toself", fillcolor="#cbd5e1",
                               line=dict(color="#475569", width=3), hoverinfo="skip"), 1, 1)
    ix, iy = retangulo(-3.2, -2.2, 3.2, 2.2)
    b.add("janela", go.Scatter(x=ix, y=iy, mode="lines", fill="toself", fillcolor="white",
                               line=dict(color="#475569", width=3), hoverinfo="skip"), 1, 1)
    px, py = espiras_xy(max(1, Np // 10), -6.4, -2.8)
    b.add("bob_p", go.Scatter(x=px, y=py, mode="lines", line=dict(color=COR_P, width=5), hoverinfo="skip"), 1, 1)
    sx, sy = espiras_xy(max(1, Ns // 10), 2.8, 6.4)
    b.add("bob_s", go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=COR_S, width=5), hoverinfo="skip"), 1, 1)

    # ---- lâmpadas e fonte ----
    if queimou:
        cor_l, simb, txt_l = "#64748b", "circle", "💥"
    else:
        cor_l, simb, txt_l = f"rgba(250,204,21,{0.10 + 0.90 * brilho:.2f})", "circle", ""
    b.add("lampadas", go.Scatter(x=xs_l, y=[0] * n_lamp, mode="markers+text", text=[txt_l] * n_lamp,
                                 textfont=dict(size=22), hoverinfo="skip",
                                 marker=dict(symbol=simb, size=34, color=cor_l, line=dict(color="#a16207", width=3))), 1, 1)
    b.add("fonte", go.Scatter(x=[-9.5], y=[0], mode="markers+text", text=["~" if fonte == "CA" else "＋ −"],
                              textfont=dict(size=22, color="#0f172a"), hoverinfo="skip",
                              marker=dict(size=48, color="white", line=dict(color="#475569", width=3))), 1, 1)
    b.add("rotulos", go.Scatter(
        x=[-4.6, 4.6, -9.5, 10.5], y=[-4.9, -4.9, -1.4, -4.6], mode="text", hoverinfo="skip",
        text=[f"<b>Primário</b><br>Np = {Np} espiras", f"<b>Secundário</b><br>Ns = {Ns} espiras",
              f"Fonte {'CA' if fonte == 'CA' else 'CC'}<br>{Vp:g} V", f"{n_lamp} lâmpada(s) 127 V / 60 W"],
        textfont=dict(size=12, color=["#c2410c", "#047857", "#334155", "#334155"])), 1, 1)

    # ---- curvas estáticas ----
    b.add("curva_vp", go.Scatter(x=t_c, y=vp_c, mode="lines", line=dict(color=COR_P, width=3),
                                 hovertemplate="t=%{x:.1f} ms<br>v_p=%{y:.1f} V<extra></extra>"), 1, 2)
    b.add("curva_vs", go.Scatter(x=t_c, y=vs_c, mode="lines", line=dict(color=COR_S, width=3),
                                 hovertemplate="t=%{x:.1f} ms<br>v_s=%{y:.1f} V<extra></extra>"), 1, 2)
    b.add("curva_phi", go.Scatter(x=t_c, y=phi_c, mode="lines", line=dict(color=COR_FLUXO, width=3),
                                  hovertemplate="t=%{x:.1f} ms<br>Φ=%{y:.2f} mWb<extra></extra>"), 2, 2)

    # ---- dinâmicos ----
    pts = perimetro(16)
    FIXOS = {
        "fluxo": (go.Scatter, 1, 1, dict(mode="markers", x=[p[0] for p in pts], y=[p[1] for p in pts], hoverinfo="skip")),
        "centro": (go.Scatter, 1, 1, dict(mode="text", x=[0], y=[0], textfont=dict(size=13, color="#0f172a"), hoverinfo="skip")),
        "txt_p": (go.Scatter, 1, 1, dict(mode="text", x=[-6], y=[5.3], textfont=dict(size=13, color="#c2410c"), hoverinfo="skip")),
        "txt_s": (go.Scatter, 1, 1, dict(mode="text", x=[6], y=[5.3], textfont=dict(size=13, color="#047857"), hoverinfo="skip")),
        "cur1": (go.Scatter, 1, 2, dict(mode="lines", y=[-v_lim, v_lim], hoverinfo="skip",
                                        line=dict(color="#64748b", width=2, dash="dot"))),
        "dot_vp": (go.Scatter, 1, 2, dict(mode="markers", hoverinfo="skip",
                                          marker=dict(size=13, color=COR_P, line=dict(color="white", width=2)))),
        "dot_vs": (go.Scatter, 1, 2, dict(mode="markers", hoverinfo="skip",
                                          marker=dict(size=13, color=COR_S, line=dict(color="white", width=2)))),
        "cur2": (go.Scatter, 2, 2, dict(mode="lines", y=[-phi_lim, phi_lim], hoverinfo="skip",
                                        line=dict(color="#64748b", width=2, dash="dot"))),
        "dot_phi": (go.Scatter, 2, 2, dict(mode="markers", hoverinfo="skip",
                                           marker=dict(size=13, color=COR_FLUXO, line=dict(color="white", width=2)))),
    }

    def dinamicos(i):
        tms, vp, vs, ph = float(t_f[i]), float(vp_f[i]), float(vs_f[i]), float(phi_f[i])
        dph = float(phi_f[min(i + 1, N_QUADROS - 1)] - phi_f[max(i - 1, 0)])
        variando = abs(dph) / phi_ref > 0.02
        mag = abs(ph) / phi_ref
        a = 0.25 + 0.75 * min(1.0, mag)
        cor = f"rgba(220,38,38,{a:.2f})" if ph >= 0 else f"rgba(37,99,235,{a:.2f})"
        sentido = "↺ anti-horário" if ph >= 0 else "↻ horário"
        estado = "fluxo VARIANDO ⇒ induz tensão" if variando else "fluxo CONSTANTE ⇒ não induz nada"
        if fonte == "CA":
            ip = reg["Ip"] * math.sqrt(2) * vp / vp_pk
            is_ = reg["Is"] * math.sqrt(2) * vs / vs_pk
            t_p = f"<b>Primário</b><br>v = {vp:+.0f} V<br>i = {ip:+.2f} A"
            t_s = f"<b>Secundário</b><br>v = {vs:+.0f} V<br>i = {is_:+.2f} A"
        else:
            t_p = f"<b>Primário</b><br>v = {vp:.0f} V (constante)"
            t_s = f"<b>Secundário</b><br>v = {vs:+.1f} V"
        return {
            "fluxo": dict(marker=dict(size=7 + 13 * min(1.0, mag), color=cor, line=dict(color="white", width=1))),
            "centro": dict(text=[f"<b>{estado}</b><br>{sentido} · {mag * 100:.0f}% do máximo"]),
            "txt_p": dict(text=[t_p]), "txt_s": dict(text=[t_s]),
            "cur1": dict(x=[tms, tms]), "dot_vp": dict(x=[tms], y=[vp]), "dot_vs": dict(x=[tms], y=[vs]),
            "cur2": dict(x=[tms, tms]), "dot_phi": dict(x=[tms], y=[ph]),
        }

    b.dinamicos(FIXOS, dinamicos(0))
    b.quadros(FIXOS, dinamicos, N_QUADROS)

    fig.update_xaxes(range=[-11.5, 14.3], visible=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[-5.8, 6.6], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(range=[0, float(t_c[-1])], gridcolor="#e2e8f0", zeroline=False, row=1, col=2)
    fig.update_yaxes(range=[-v_lim, v_lim], gridcolor="#e2e8f0", zerolinecolor="#94a3b8", title_text="tensão (V)", row=1, col=2)
    fig.update_xaxes(range=[0, float(t_c[-1])], gridcolor="#e2e8f0", zeroline=False, title_text="tempo (ms)", row=2, col=2)
    fig.update_yaxes(range=[-phi_lim, phi_lim], gridcolor="#e2e8f0", zerolinecolor="#94a3b8", title_text="Φ (mWb)", row=2, col=2)
    fig.update_layout(height=760, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=90, b=10),
                      updatemenus=botoes_animacao(dur_ms, "▶ Ligar transformador"),
                      sliders=slider_frames([f"{t:.1f}" for t in t_f], "⏱ t = ", " ms  (arraste para ver quadro a quadro)"))
    return fig


def gerar_balanco_potencia(reg):
    fig = go.Figure(go.Bar(
        x=["Entrada (rede)", "Saída (lâmpadas)", "Perdas (calor)"], y=[reg["Pin"], reg["Ps"], reg["perdas"]],
        marker_color=[COR_P, COR_S, "#ef4444"], text=[f"{v:.0f} W" for v in (reg["Pin"], reg["Ps"], reg["perdas"])],
        textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      title=dict(text="Balanço de potência (conservação de energia)", font=dict(size=14)),
                      yaxis=dict(title="potência (W)", gridcolor="#e2e8f0", rangemode="tozero"))
    return fig


# ============================================
# FIGURA 2 — BOBINA DE TESLA
# ============================================
def gerar_animacao_tesla(d, V0_kV, dur_ms):
    N_Q = 80
    ix = np.linspace(0, len(d["t_us"]) - 1, N_Q).astype(int)
    sl = slice(None, None, 4)
    t_us, V1, V2, e1, e2, fs = d["t_us"], d["V1"], d["V2"], d["env1"], d["env2"], d["fs"]
    lim1, lim2 = 1.12 * V0_kV, 1.12 * max(float(np.max(np.abs(V2))), 1.0)
    fs_max = float(fs.max())

    # faíscas: 4 variantes aleatórias (para "tremer") de 6 polilinhas
    rng = np.random.default_rng(7)
    x0s = np.array([-1.5, -1.0, -0.4, 0.4, 1.0, 1.5])
    y0s = 10.2 + 0.5 * np.sqrt(1 - (x0s / 1.6) ** 2)
    thetas = [-1.2, -0.7, -0.2, 0.2, 0.7, 1.2]
    fator = [1.0, 0.8, 0.95, 0.7, 0.9, 0.75]
    variantes = []
    for _ in range(4):
        v = []
        for th in thetas:
            ang = th + rng.normal(0, 0.45, 14)
            v.append((np.concatenate([[0], np.cumsum(np.sin(ang)) / 14]), np.concatenate([[0], np.cumsum(np.cos(ang)) / 14])))
        variantes.append(v)

    fig = make_subplots(rows=3, cols=2, column_widths=[0.5, 0.5], row_heights=[0.26, 0.37, 0.37],
                        specs=[[{"rowspan": 3}, {}], [None, {}], [None, {}]],
                        horizontal_spacing=0.08, vertical_spacing=0.10,
                        subplot_titles=("Bobina de Tesla (esquema)", "Onde está a energia?",
                                        "Tensão no capacitor primário (kV)", "Tensão no topo do secundário (kV)"))
    b = Construtor(fig)

    # ---- esquema estático ----
    sx, sy = retangulo(-0.9, 1.5, 0.9, 9.0)
    b.add("sec_corpo", go.Scatter(x=sx, y=sy, mode="lines", fill="toself", fillcolor="rgba(253,230,138,0.35)",
                                  line=dict(color="#78350f", width=3), hoverinfo="skip"), 1, 1)
    tx, ty = [], []
    for y in np.linspace(1.7, 8.8, 30):
        tx += [-0.9, 0.9, None]
        ty += [y, y, None]
    b.add("sec_espiras", go.Scatter(x=tx, y=ty, mode="lines", line=dict(color="#b45309", width=1.5), hoverinfo="skip"), 1, 1)
    b.add("fios", go.Scatter(
        x=[0, 0, None, -0.7, 0.7, None, -0.4, 0.4, None, 0, 0, None,
           -3.0, -3.0, -1.45, None, -0.95, 1.15, None, 1.45, 3.0, 3.0],
        y=[9.0, 9.7, None, 0, 0, None, -0.25, -0.25, None, 0, 1.5, None,
           1.0, -2.5, -2.5, None, -2.5, -2.5, None, -2.5, -2.5, 1.0],
        mode="lines", line=dict(color="#475569", width=3), hoverinfo="skip"), 1, 1)
    b.add("capacitor", go.Scatter(x=[-1.45, -1.45, None, -0.95, -0.95], y=[-3.2, -1.8, None, -3.2, -1.8], mode="lines",
                                  line=dict(color="#0f172a", width=6), hoverinfo="skip"), 1, 1)
    b.add("centelhador", go.Scatter(x=[1.15, 1.15, None, 1.45, 1.45, None, 1.15, 1.25, 1.35, 1.45],
                                    y=[-2.9, -2.1, None, -2.9, -2.1, None, -2.5, -2.2, -2.8, -2.5], mode="lines",
                                    line=dict(color="#a855f7", width=3), hoverinfo="skip"), 1, 1)
    b.add("rotulos", go.Scatter(
        x=[1.2, -3.4, -1.2, 1.3, 2.0], y=[5.5, 1.0, -3.9, -3.9, 10.9], mode="text", hoverinfo="skip",
        textposition=["middle right", "middle left", "middle center", "middle center", "middle right"],
        text=["<b>Secundário</b><br>(milhares de espiras)", "<b>Primário</b><br>(poucas espiras)",
              "Capacitor C₁", "Centelhador", "Toroide (C₂)"],
        textfont=dict(size=12, color="#334155")), 1, 1)

    # ---- curvas estáticas ----
    b.add("v1", go.Scatter(x=t_us[sl], y=V1[sl], mode="lines", line=dict(color="#fdba74", width=1.5), hoverinfo="skip"), 2, 2)
    b.add("v1_env", go.Scatter(x=np.concatenate([t_us[sl], [None], t_us[sl]]), y=np.concatenate([e1[sl], [None], -e1[sl]]),
                               mode="lines", line=dict(color=COR_P, width=2, dash="dash"), hoverinfo="skip"), 2, 2)
    b.add("v2", go.Scatter(x=t_us[sl], y=V2[sl], mode="lines", line=dict(color="#6ee7b7", width=1.5), hoverinfo="skip"), 3, 2)
    b.add("v2_env", go.Scatter(x=np.concatenate([t_us[sl], [None], t_us[sl]]), y=np.concatenate([e2[sl], [None], -e2[sl]]),
                               mode="lines", line=dict(color="#059669", width=2, dash="dash"), hoverinfo="skip"), 3, 2)

    # ---- dinâmicos ----
    ang = np.linspace(0, 2 * np.pi, 60)
    ring_x, ring_y = 1.6 * np.cos(ang), 10.2 + 0.5 * np.sin(ang)
    prim_x = [-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0]
    FIXOS = {
        "toroide": (go.Scatter, 1, 1, dict(mode="lines", x=ring_x, y=ring_y, fill="toself",
                                           line=dict(color="#64748b", width=3), hoverinfo="skip")),
        "faiscas": (go.Scatter, 1, 1, dict(mode="lines", line=dict(color="#c084fc", width=2.5), hoverinfo="skip")),
        "prim_marc": (go.Scatter, 1, 1, dict(mode="markers", x=prim_x, y=[1.0] * 8, hoverinfo="skip")),
        "legenda": (go.Scatter, 1, 1, dict(mode="text", x=[0], y=[-5.4], textfont=dict(size=13, color="#0f172a"), hoverinfo="skip")),
        "barras": (go.Bar, 1, 2, dict(x=["Primário", "Secundário"], marker_color=[COR_P, COR_S],
                                      textposition="outside", hoverinfo="skip")),
        "cur_v1": (go.Scatter, 2, 2, dict(mode="lines", y=[-lim1, lim1], hoverinfo="skip",
                                          line=dict(color="#64748b", width=2, dash="dot"))),
        "dot_v1": (go.Scatter, 2, 2, dict(mode="markers", hoverinfo="skip",
                                          marker=dict(size=12, color=COR_P, line=dict(color="white", width=2)))),
        "cur_v2": (go.Scatter, 3, 2, dict(mode="lines", y=[-lim2, lim2], hoverinfo="skip",
                                          line=dict(color="#64748b", width=2, dash="dot"))),
        "dot_v2": (go.Scatter, 3, 2, dict(mode="markers", hoverinfo="skip",
                                          marker=dict(size=12, color=COR_S, line=dict(color="white", width=2)))),
    }

    def dinamicos(i):
        j = int(ix[i])
        tus, ee1, ee2, share = float(t_us[j]), float(e1[j]), float(e2[j]), float(fs[j])
        subindo = share >= float(fs[int(ix[max(i - 1, 0)])])
        if fs_max < 35:
            fase = "⚠ Fora de sintonia: quase nenhuma energia chega ao secundário"
        elif share < 8 and subindo:
            fase = "① Energia guardada no primário (capacitor carregado)"
        elif share >= 75:
            fase = "③ Quase toda a energia no secundário: TENSÃO MÁXIMA!"
        elif subindo:
            fase = "② Energia fluindo do primário para o secundário"
        else:
            fase = "④ Energia voltando ao primário (vai e vem)"
        comp = 4.3 * float(np.clip((ee2 - 30.0) / (420.0 - 30.0), 0, 1))
        if comp < 0.15:
            fx, fy = [None] * 6 * 16, [None] * 6 * 16
        else:
            fx, fy = [], []
            for s, (px, py) in enumerate(variantes[i % 4]):
                L = comp * fator[s]
                fx += list(x0s[s] + L * px) + [None]
                fy += list(y0s[s] + L * py) + [None]
        brilho = 0.15 + 0.75 * float(np.clip(ee2 / 400.0, 0, 1))
        pa = 0.25 + 0.75 * (100 - share) / 100
        return {
            "toroide": dict(fillcolor=f"rgba(251,191,36,{brilho:.2f})"),
            "faiscas": dict(x=fx, y=fy),
            "prim_marc": dict(marker=dict(size=13, color=f"rgba(249,115,22,{pa:.2f})", line=dict(color="#7c2d12", width=1.5))),
            "legenda": dict(text=[f"<b>{fase}</b><br>V₂ (pico local) ≈ {ee2:.0f} kV — o comprimento das faíscas acompanha V₂"]),
            "barras": dict(y=[100 - share, share], text=[f"{100 - share:.0f}%", f"{share:.0f}%"]),
            "cur_v1": dict(x=[tus, tus]), "dot_v1": dict(x=[tus], y=[ee1]),
            "cur_v2": dict(x=[tus, tus]), "dot_v2": dict(x=[tus], y=[ee2]),
        }

    b.dinamicos(FIXOS, dinamicos(0))
    b.quadros(FIXOS, dinamicos, N_Q)

    fig.update_xaxes(range=[-7.5, 7.5], visible=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[-6.4, 14.8], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(range=[0, 118], title_text="% da energia", gridcolor="#e2e8f0", row=1, col=2)
    for r_, lim in ((2, lim1), (3, lim2)):
        fig.update_xaxes(range=[0, float(t_us[-1])], gridcolor="#e2e8f0", zeroline=False,
                         title_text="tempo (µs)" if r_ == 3 else None, row=r_, col=2)
        fig.update_yaxes(range=[-lim, lim], gridcolor="#e2e8f0", zerolinecolor="#94a3b8", row=r_, col=2)
    fig.update_layout(height=780, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=90, b=10),
                      updatemenus=botoes_animacao(dur_ms, "▶ Disparar a bobina"),
                      sliders=slider_frames([f"{t_us[k]:.1f}" for k in ix], "⏱ t = ", " µs  (arraste para ver quadro a quadro)"))
    return fig


def gerar_curva_ressonancia(V0_kV, k, perdas, f1_atual, v2_atual):
    f1, v2 = varredura_ressonancia(V0_kV, k, perdas)
    fig = go.Figure(go.Scatter(x=f1, y=v2, mode="lines+markers", line=dict(color="#3b82f6", width=3),
                               hovertemplate="f₁=%{x:.0f} kHz<br>V₂ máx=%{y:.0f} kV<extra></extra>"))
    fig.add_vline(x=F2 / 1e3, line=dict(color="#10b981", dash="dash", width=2),
                  annotation_text="f₂ do secundário", annotation_position="top")
    fig.add_trace(go.Scatter(x=[f1_atual], y=[v2_atual], mode="markers", hoverinfo="skip",
                             marker=dict(size=15, color=COR_P, line=dict(color="white", width=2))))
    fig.update_layout(height=320, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10),
                      title=dict(text="Curva de ressonância: tensão máxima no secundário × frequência do primário", font=dict(size=14)),
                      xaxis=dict(title="frequência do circuito primário f₁ (kHz)", gridcolor="#e2e8f0"),
                      yaxis=dict(title="V₂ máxima (kV)", gridcolor="#e2e8f0", rangemode="tozero"))
    return fig


# ============================================
# FIGURA 3 — TRANSMISSÃO DE ENERGIA
# ============================================
def gerar_figura_transmissao(res, P, V, dist, r, dur_ms):
    N_F, CICLOS, N_PT = 45, 3, 10
    x_ini, x_fim = 5.2, 15.0
    frac = float(np.clip(res["pct"] / 20.0, 0, 1)) if res["viavel"] else 1.0
    cor_lin = f"rgb({int(34 + 205 * frac)},{int(197 - 129 * frac)},{int(94 - 26 * frac)})"
    I = res["I"] if res["viavel"] else P / V * 1000
    tam = 5 + 20 * min(1.0, I / 3000.0)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.08,
                        subplot_titles=("Da usina à cidade", "Perda na linha × tensão de transmissão"))
    b = Construtor(fig)

    ux, uy = retangulo(0, -1.2, 2.4, 1.2)
    b.add("usina", go.Scatter(x=ux, y=uy, mode="lines", fill="toself", fillcolor="#e0f2fe", line=dict(color="#0369a1", width=3),
                              hoverinfo="skip"), 1, 1)
    xb, yb, xc, yc = [], [], [], []
    for cx in (3.9, 4.6, 15.6, 16.3):
        x_, y_ = circulo(cx, 0, 0.6)
        (xb if cx in (3.9, 15.6) else xc).extend(x_ + [None])
        (yb if cx in (3.9, 15.6) else yc).extend(y_ + [None])
    b.add("trafo_a", go.Scatter(x=xb, y=yb, mode="lines", line=dict(color=COR_P, width=4), hoverinfo="skip"), 1, 1)
    b.add("trafo_b", go.Scatter(x=xc, y=yc, mode="lines", line=dict(color=COR_S, width=4), hoverinfo="skip"), 1, 1)
    b.add("fios_bt", go.Scatter(x=[2.4, 3.3, None, 16.9, 18.0],
                                y=[0, 0, None, 0, 0], mode="lines",
                                line=dict(color="#475569", width=3), hoverinfo="skip"), 1, 1)
    b.add("linha", go.Scatter(x=[x_ini, x_fim], y=[0, 0], mode="lines", line=dict(color=cor_lin, width=7), hoverinfo="skip"), 1, 1)
    px, py = [], []
    for x in (6.6, 9.4, 12.2):
        px += [x, x, None, x - 0.5, x + 0.5, None]
        py += [-1.7, 0, None, -0.3, -0.3, None]
    b.add("torres", go.Scatter(x=px, y=py, mode="lines", line=dict(color="#94a3b8", width=3), hoverinfo="skip"), 1, 1)
    cx_, cy_ = [], []
    for x0, alt in ((18.0, 1.6), (19.2, 2.6), (20.4, 1.2)):
        rx, ry = retangulo(x0, -1.2, x0 + 0.9, -1.2 + alt)
        cx_ += rx + [None]
        cy_ += ry + [None]
    b.add("cidade", go.Scatter(x=cx_, y=cy_, mode="lines", fill="toself", fillcolor="#fef9c3",
                               line=dict(color="#a16207", width=2), hoverinfo="skip"), 1, 1)
    aviso = (f"perda: <b>{res['perda']:.1f} MW ({res['pct']:.1f}%)</b> 🔥" if res["viavel"]
             else "❌ <b>Impossível:</b> a linha não entrega essa potência")
    b.add("textos", go.Scatter(
        x=[1.2, 4.25, 4.25, 15.95, 15.95, 19.3, 10.1, 10.1], y=[0, 1.5, -1.5, 1.5, -1.5, -1.9, 1.9, -2.5], mode="text",
        hoverinfo="skip", text=["🏭<br>Usina", "13,8 kV → " + f"{V:g} kV", "Elevador", f"{V:g} kV → 127 V", "Abaixador",
                                f"Cidade<br>{P:g} MW", f"<b>{V:g} kV</b> · corrente ≈ {I:,.0f} A".replace(",", "."), aviso],
        textfont=dict(size=12, color="#0f172a")), 1, 1)

    spacing = (x_fim - x_ini) / N_PT
    FIXOS = {"cargas": (go.Scatter, 1, 1, dict(mode="markers", y=[0] * N_PT, hoverinfo="skip",
                                              marker=dict(size=tam, color="#facc15", line=dict(color="#a16207", width=1.5))))}

    def dinamicos(f):
        ph = (f % N_F) / N_F
        return {"cargas": dict(x=[x_ini + ((j + ph) % N_PT) * spacing for j in range(N_PT)])}

    b.dinamicos(FIXOS, dinamicos(0))
    b.quadros(FIXOS, dinamicos, N_F * CICLOS)

    Vs = np.geomspace(10, 800, 120)
    curva = [calc_transmissao(P, dist, v, r) for v in Vs]
    pct = [c["pct"] if c["viavel"] else np.nan for c in curva]
    b.add("curva", go.Scatter(x=Vs, y=pct, mode="lines", line=dict(color="#3b82f6", width=3),
                              hovertemplate="V=%{x:.0f} kV<br>perda=%{y:.1f}%<extra></extra>"), 1, 2)
    if res["Vmin"] > 10:
        fig.add_vrect(x0=10, x1=res["Vmin"], fillcolor="rgba(239,68,68,0.12)", line_width=0, row=1, col=2)
    if res["viavel"]:
        b.add("agora", go.Scatter(x=[V], y=[res["pct"]], mode="markers", hoverinfo="skip",
                                  marker=dict(size=15, color=cor_lin, line=dict(color="white", width=2))), 1, 2)

    fig.update_xaxes(range=[-0.5, 21.8], visible=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[-3.2, 3.2], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(type="log", range=[1, math.log10(800)], title_text="tensão de transmissão (kV)", gridcolor="#e2e8f0", row=1, col=2)
    fig.update_yaxes(range=[0, 60], title_text="perda (% da potência enviada)", gridcolor="#e2e8f0", row=1, col=2)
    fig.update_layout(height=430, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=90, b=10), updatemenus=botoes_animacao(dur_ms, "▶ Fluxo de corrente"))
    return fig


# ============================================
# BARRA LATERAL
# ============================================
with st.sidebar:
    st.header("⚙️ Configurações")
    avancado = st.checkbox("Mostrar equações e valores numéricos", value=False,
                           help="Desligado: foco na intuição. Ligado: mostra as equações usadas em cada simulação.")
    st.markdown("---")
    st.subheader("🧑‍🏫 Roteiro sugerido")
    st.markdown(
        "**1. Prever** – peça que os alunos apostem antes de rodar (caixas *Teste sua intuição*).\n\n"
        "**2. Observar** – rodem a animação e **arrastem a barra de tempo** para pausar em qualquer instante.\n\n"
        "**3. Explicar** – *o que variou? o que se conservou? onde foi parar a energia?*"
    )
    st.markdown("---")
    st.caption("Cores: 🟠 primário · 🟢 secundário · 🔵 fluxo magnético")

st.markdown('<div class="main-title">⚡ Transformador, Bobina de Tesla e Rede Elétrica</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Uma mesma ideia — indução mútua — em três escalas: da tomada ao raio</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["1. Transformador", "2. Bobina de Tesla", "3. Transmissão de energia"])

# ---------------- ABA 1: TRANSFORMADOR ----------------
with tab1:
    st.markdown("""
    <div class="concept-card">
        <b>Transformador:</b> duas bobinas enroladas no mesmo núcleo de ferro. A corrente <i>alternada</i> no
        <span style="color:#c2410c"><b>primário</b></span> cria um fluxo magnético que <b>varia</b> no núcleo; esse fluxo
        atravessa o <span style="color:#047857"><b>secundário</b></span> e, pela Lei de Faraday, induz uma tensão.
        Quem manda no resultado é a <b>razão entre os números de espiras</b>.
    </div>
    """, unsafe_allow_html=True)
    with st.expander("💡 Analogia: a marcha da bicicleta"):
        st.markdown(
            "Pense numa **bicicleta com marchas**: marcha pesada = muita força e pouca velocidade; marcha leve = o contrário. "
            "Mas a **potência das suas pernas** é a mesma. No transformador, *tensão* faz o papel de velocidade e *corrente* "
            "o de força: se o secundário tem **mais espiras**, a tensão sobe e a corrente cai — **a potência não muda** "
            "(no ideal). Ninguém ganha energia de graça."
        )

    ct1, ct2 = st.columns([1, 2.7])
    with ct1:
        with st.container(border=True):
            st.markdown("**Controles**")
            fonte_lbl = st.radio("Fonte no primário", ["🔌 Corrente alternada (CA)", "🔋 Corrente contínua (CC)"], key="fonte_t")
            fonte = "CA" if "CA" in fonte_lbl else "CC"
            Vp = st.radio("Tensão da fonte (V)", [127, 220], horizontal=True, key="vp_t")
            Np = st.slider("Espiras do primário (Np)", 10, 300, 100, 10, key="np_t")
            Ns = st.slider("Espiras do secundário (Ns)", 10, 300, 100, 10, key="ns_t")
            n_lamp = st.slider("Lâmpadas no secundário (em paralelo)", 1, 5, 1, key="nl_t")
            eta = st.slider("Rendimento do transformador (%)", 70, 100, 100, key="eta_t",
                            help="100% = ideal. Menos que isso = perdas por calor no cobre e no ferro.") / 100
            dur_t = st.select_slider("Velocidade da animação", options=[120, 60, 30], value=60, key="dur_t",
                                     format_func=lambda x: {120: "Lenta", 60: "Normal", 30: "Rápida"}[x])
            st.caption("A animação roda em câmera lenta: 60 Hz = 60 ciclos por segundo!")

        razao = Ns / Np
        tipo = "ELEVADOR ⬆" if razao > 1 else ("ABAIXADOR ⬇" if razao < 1 else "1:1 (isolador)")
        st.info(f"Relação Ns/Np = **{razao:.2f}** → transformador **{tipo}**")
        with st.expander("🔬 Desafios para explorar"):
            st.markdown(
                "1. **Ns = 2·Np.** O que acontece com a tensão? E com a lâmpada?\n"
                "2. **Ns = Np/2.** Ela brilha mais ou menos? Por quê?\n"
                "3. **Some lâmpadas.** A corrente do *primário* muda mesmo sem mexer no primário?\n"
                "4. **Troque para CC.** O que a lâmpada faz? Olhe o fluxo no gráfico.\n"
                "5. **Reduza o rendimento.** Para onde vai a energia que não chega à lâmpada?"
            )

    reg = regime_transformador(Np, Ns, Vp, n_lamp, eta)
    with ct2:
        if fonte == "CA":
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tensão no secundário", f"{reg['Vs']:.0f} V")
            m2.metric("Corrente no secundário", f"{reg['Is']:.2f} A")
            m3.metric("Corrente no primário", f"{reg['Ip']:.2f} A")
            m4.metric("Potência nas lâmpadas", f"{reg['Ps']:.0f} W")
            if reg["Vs"] > 1.3 * V_LAMP:
                st.error("💥 A tensão no secundário é alta demais: as lâmpadas de 127 V queimaram!")
            elif reg["Vs"] < 0.6 * V_LAMP:
                st.warning("🔅 Tensão baixa demais: as lâmpadas mal acendem.")
        else:
            st.error("🔋 Com CC o fluxo para de variar ⇒ **sem tensão induzida**: as lâmpadas ficam apagadas. "
                     "E há um perigo real: um primário ligado direto a uma bateria só tem a resistência do fio "
                     "limitando a corrente — ele superaquece! (Por isso a rede usa CA.)")
        mostrar(gerar_animacao_transformador(fonte, Np, Ns, Vp, n_lamp, eta, dur_t))

    if fonte == "CA":
        mostrar(gerar_balanco_potencia(reg))
    if avancado:
        st.latex(r"\frac{V_s}{V_p}=\frac{N_s}{N_p}\qquad \frac{I_p}{I_s}=\frac{N_s}{N_p}\qquad P_{entrada}=\frac{P_{saída}}{\eta}")
        st.latex(r"v = N\,\frac{d\Phi}{dt}\;\Rightarrow\;\Phi_{max}=\frac{V_{pico}}{N\,\omega}")
        st.caption(f"Aqui: Φ_max ≈ {Vp * math.sqrt(2) / (Np * 2 * math.pi * FREQ_REDE) * 1e3:.2f} mWb. "
                   "Mais espiras no primário ⇒ menos fluxo necessário no núcleo para a mesma tensão.")

    pergunta("trafo1", "Um transformador elevador (Ns > Np) aumenta a tensão. O que acontece com a corrente no secundário (ideal)?",
             ["Aumenta também", "Diminui, pois a potência se conserva", "Não muda"], 1,
             "P = V·I se conserva no ideal: se V sobe, I desce na mesma proporção.")
    pergunta("trafo2", "Por que o transformador não funciona ligado a uma pilha (CC)?",
             ["A pilha tem pouca energia", "O fluxo no núcleo fica constante e a Lei de Faraday exige VARIAÇÃO",
              "O ferro não conduz corrente contínua"], 1,
             "Sem variação de fluxo não há tensão induzida. Só no instante de ligar/desligar aparece um pulso.")

# ---------------- ABA 2: TESLA ----------------
with tab2:
    st.markdown("""
    <div class="concept-card" style="border-left-color:#a855f7;">
        <b>Bobina de Tesla:</b> é um <b>transformador ressonante</b>. Um capacitor carregado (primário) descarrega através de um
        centelhador e faz o circuito primário <i>oscilar</i>. O secundário — uma bobina alta, sem ferro — é <b>sintonizado
        na mesma frequência</b>. A cada oscilação o primário "empurra" o secundário no ritmo certo, e a energia passa de um para
        o outro até que, no topo, a tensão chega a centenas de milhares de volts.
    </div>
    """, unsafe_allow_html=True)
    with st.expander("💡 Analogia: empurrar o balanço + dois pêndulos"):
        st.markdown(
            "**Balanço no parque:** pequenos empurrões dados **no ritmo certo** (na frequência natural do balanço) fazem a amplitude "
            "crescer muito — isso é *ressonância*. Um empurrão fora do ritmo não faz quase nada.\n\n"
            "**Dois pêndulos ligados por uma mola:** solte um deles e a energia vai passando para o outro, e depois volta — "
            "o vai e vem do gráfico *Onde está a energia?*. A **mola** é o acoplamento magnético entre as bobinas (k)."
        )

    cs1, cs2 = st.columns([1, 2.7])
    with cs1:
        with st.container(border=True):
            st.markdown("**Ajustes da bobina**")
            if "c1_tesla" not in st.session_state:
                st.session_state["c1_tesla"] = round(C1_SINTONIA * 1e9, 1)
            c1_nF = st.slider("Capacitor primário C₁ (nF)", 30.0, 60.0, step=0.5, key="c1_tesla")
            st.button("🎯 Sintonizar automaticamente", key="btn_sint",
                      on_click=lambda: st.session_state.update(c1_tesla=round(C1_SINTONIA * 1e9, 1)))
            V0 = st.slider("Tensão de carga do capacitor (kV)", 5.0, 15.0, 10.0, 0.5, key="v0_tesla")
            k_ac = st.slider("Acoplamento entre as bobinas (k)", 0.05, 0.35, 0.15, 0.01, key="k_tesla",
                             help="Fração do fluxo do primário que atravessa o secundário. Transformador de ferro: k ≈ 1.")
            perdas = st.slider("Perdas (resistência do fio, calor…)", 0.5, 3.0, 1.0, 0.5, key="perd_tesla")
            dur_ts = st.select_slider("Velocidade da animação", options=[150, 80, 40], value=80, key="dur_ts",
                                      format_func=lambda x: {150: "Lenta", 80: "Normal", 40: "Rápida"}[x])
            st.caption("A simulação cobre 100 µs (microssegundos!) — a animação é câmera ultralenta.")
        with st.expander("🔬 Desafios para explorar"):
            st.markdown(
                "1. **Afaste C₁ da sintonia (30 ou 60 nF)** e veja a tensão do topo e a energia transferida caírem (curva de ressonância abaixo).\n"
                "2. **Sintonize e aumente k.** A energia passa mais rápido? E se k for enorme?\n"
                "3. **Aumente as perdas.** Quem perde mais tensão, o pico ou a duração?\n"
                "4. **Dobre a tensão do capacitor.** O que acontece com o comprimento das faíscas?"
            )

    d = dados_tesla(float(c1_nF), float(V0), float(k_ac), float(perdas))
    f1 = 1 / (2 * math.pi * math.sqrt(L1 * c1_nF * 1e-9)) / 1e3
    dfp = (f1 * 1e3 - F2) / F2 * 100
    sint = "✅ sintonizado" if abs(dfp) < 1.5 else ("🟡 quase" if abs(dfp) < 5 else "❌ fora de sintonia")
    with cs2:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Frequência do primário", f"{f1:.0f} kHz", f"{dfp:+.1f}% vs secundário", delta_color="off")
        q2.metric("Sintonia", sint)
        q3.metric("Tensão máxima no topo", f"{d['V2max']:.0f} kV")
        q4.metric("Energia transferida (máx.)", f"{float(d['fs'].max()):.0f}%")
        mostrar(gerar_animacao_tesla(d, float(V0), dur_ts))

    mostrar(gerar_curva_ressonancia(float(V0), float(k_ac), float(perdas), f1, d["V2max"]))

    st.markdown("#### ⚖️ Transformador comum × Bobina de Tesla")
    st.markdown(
        "| | Transformador de ferro | Bobina de Tesla |\n|---|---|---|\n"
        "| Núcleo | ferro (guia o fluxo) | ar |\n"
        "| Acoplamento k | ≈ 1 (forte) | ≈ 0,1 – 0,2 (fraco) |\n"
        "| Ressonância | não usa | **essencial** (primário e secundário sintonizados) |\n"
        "| Como ganha tensão | razão de espiras Ns/Np | energia oscilando e se acumulando no secundário |\n"
        "| Corrente | contínua senoidal (60 Hz) | oscilação amortecida (~150 kHz) |"
    )
    if avancado:
        st.latex(r"f=\frac{1}{2\pi\sqrt{LC}}\qquad k=\frac{M}{\sqrt{L_1L_2}}\qquad "
                 r"\tfrac12 C_1V_1^2=\tfrac12 C_2V_2^2\;\Rightarrow\;V_2\approx V_1\sqrt{\tfrac{L_2}{L_1}}\ \text{(sintonizado, sem perdas)}")
        st.caption(f"Aqui: L₁ = {L1 * 1e6:.0f} µH, L₂ = {L2 * 1e3:.0f} mH, C₂ ≈ {C2 * 1e12:.1f} pF ⇒ f₂ = {F2 / 1e3:.0f} kHz; "
                   f"√(L₂/L₁) ≈ {math.sqrt(L2 / L1):.0f}. Tempo de transferência total ≈ 1/(2·k·f) ≈ {1e6 / (2 * k_ac * F2):.0f} µs.")

    pergunta("tesla1", "Por que a Tesla precisa que primário e secundário tenham a MESMA frequência natural?",
             ["Para o metal aquecer menos", "Para a energia passar de um para o outro de forma eficiente (ressonância)",
              "Porque só assim existe campo magnético"], 1,
             "Como o balanço: só empurrões no ritmo certo acumulam energia. Fora de sintonia, quase nada passa.")
    pergunta("tesla2", "A tensão no topo é muito maior que a do capacitor. De onde vem a energia extra?",
             ["Ela é criada pela bobina", "Não há energia extra: a energia do capacitor fica concentrada em capacitância muito menor",
              "Do ar ao redor"], 1,
             "Conserva-se ½CV². Como C₂ é ~2000× menor que C₁, V₂ precisa ser ~45× maior (√2000).")

# ---------------- ABA 3: TRANSMISSÃO ----------------
with tab3:
    st.markdown("""
    <div class="alert-card">
        <b>Por que a rede elétrica usa alta tensão?</b> Os fios têm resistência. A perda por aquecimento é
        <b>P = R·I²</b> — depende do <i>quadrado da corrente</i>. Para entregar a mesma potência (P = V·I), aumentar a
        tensão <b>reduz a corrente</b>. E quem eleva e abaixa a tensão com eficiência? O <b>transformador</b> — que só
        funciona em CA. Foi assim que a CA de Tesla e Westinghouse venceu a CC de Edison na "Guerra das Correntes".
    </div>
    """, unsafe_allow_html=True)
    with st.expander("💡 Analogia: caminhões × carrinhos de mão"):
        st.markdown(
            "Para levar 100 toneladas de arroz até a cidade, você pode usar **milhares de carrinhos de mão** (baixa tensão, "
            "corrente enorme) ou **poucos caminhões gigantes** (alta tensão, corrente pequena). Os carrinhos entopem a estrada "
            "e muito arroz se perde no caminho; com os caminhões, quase nada. A 'estrada entupida' é o calor nos fios."
        )

    cx1, cx2 = st.columns([1, 2.7])
    with cx1:
        with st.container(border=True):
            st.markdown("**Parâmetros da linha**")
            P_MW = st.slider("Potência entregue à cidade (MW)", 10, 500, 100, 10, key="p_tr")
            dist = st.slider("Distância usina–cidade (km)", 50, 1000, 300, 50, key="d_tr")
            V_kV = st.select_slider("Tensão de transmissão (kV)", options=[13.8, 34.5, 69, 138, 230, 345, 500, 750],
                                    value=230, key="v_tr")
            r_km = st.slider("Resistência do cabo (Ω/km)", 0.02, 0.20, 0.08, 0.01, key="r_tr",
                             help="Cabo mais grosso = resistência menor.")
            dur_tr = st.select_slider("Velocidade da animação", options=[120, 60, 30], value=60, key="dur_tr",
                                      format_func=lambda x: {120: "Lenta", 60: "Normal", 30: "Rápida"}[x])
        with st.expander("🔬 Desafios para explorar"):
            st.markdown(
                "1. **Baixe a tensão para 13,8 kV.** O que acontece com o tamanho das bolinhas (corrente)?\n"
                "2. **Dobre a tensão** (ex.: 115→230). Quantas vezes a perda diminui?\n"
                "3. **Aumente a distância.** Como compensar sem mudar o cabo?\n"
                "4. **Cabo mais grosso** (menor Ω/km) vs **tensão mais alta**: qual resolve mais?"
            )

    res = calc_transmissao(P_MW, dist, V_kV, r_km)
    with cx2:
        if res["viavel"]:
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Corrente na linha", f"{res['I']:,.0f} A".replace(",", "."))
            n2.metric("Perda por aquecimento", f"{res['perda']:.1f} MW")
            n3.metric("Perda relativa", f"{res['pct']:.1f}%")
            n4.metric("Potência gerada", f"{res['Pe']:.1f} MW")
        else:
            st.error(f"❌ Com {V_kV:g} kV a linha não consegue entregar {P_MW} MW: a corrente necessária derreteria os cabos. "
                     f"Seria preciso pelo menos ≈ {res['Vmin']:.0f} kV.")
        mostrar(gerar_figura_transmissao(res, P_MW, V_kV, dist, r_km, dur_tr))

    if avancado:
        st.latex(r"I=\frac{P}{V}\qquad P_{perdida}=R\,I^2=\frac{R\,P^2}{V^2}\qquad R=r\cdot d")
        st.caption("Dobrar V ⇒ I cai à metade ⇒ perda cai a ≈ 1/4 (um pouco menos, porque a usina precisa gerar também o que se perde). Modelo simplificado (linha resistiva, monofásica equivalente; "
                   "ignora reatância e efeito corona).")
    if res["viavel"]:
        meia = calc_transmissao(P_MW, dist, V_kV / 2, r_km)
        if meia["viavel"]:
            st.info(f"Com metade da tensão ({V_kV / 2:g} kV) a perda seria **{meia['perda']:.1f} MW** "
                    f"({meia['perda'] / max(res['perda'], 1e-9):.1f}× maior).")

    pergunta("rede", "Se dobrarmos a tensão de transmissão mantendo a mesma potência entregue, a perda por aquecimento…",
             ["Cai à metade", "Cai a um quarto", "Não muda"], 1,
             "A corrente cai à metade (I = P/V) e a perda depende de I² ⇒ 1/4.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#94a3b8;font-size:0.85rem;padding:1rem;">'
            '⚡ <b>Física Visual</b> — indução mútua do transformador à Tesla e à rede elétrica.</div>',
            unsafe_allow_html=True)
