import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io, base64

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def trend_chart(df):
    daily = df.groupby("date")["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(daily["date"], daily["sales"], color="#4E79A7", linewidth=1.5)
    ax.set_title("Total Daily Sales — 2022–2023", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Sales")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)

def seasonality_chart(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = df.groupby("day_of_week")["sales"].mean().reindex(order)
    colors = ["#4E79A7"]*5 + ["#59A14F"]*2
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(dow.index, dow.values, color=colors)
    ax.set_title("Average Sales by Day of Week", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Sales")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)

def store_product_chart(df):
    grp = df.groupby(["store_id","product_id"])["sales"].mean().reset_index()
    labels = [f"S{int(r.store_id)}-P{int(r.product_id)}" for _,r in grp.iterrows()]
    colors = ["#4E79A7","#59A14F","#F28E2B","#E15759","#9467BD","#76B7B2"]
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(labels, grp["sales"].values, color=colors[:len(labels)])
    ax.set_title("Average Daily Sales by Store & Product", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Sales"); ax.set_xlabel("Store-Product")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)

def generate_all_charts(df):
    return {
        "trend": trend_chart(df),
        "seasonality": seasonality_chart(df),
        "store_product": store_product_chart(df),
    }
