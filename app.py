import matplotlib.pyplot as plt
import solara
import pandas as pd
import mesa
from model import TransportModel
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
)

# Fixed parameters
NUM_AGENTS = 2000000
CAR_COST = 5.0
BUS_COST = 2.0
TRAIN_COST = 3.0
LAMBDA_PRIVATE = 0.8
LAMBDA_PUBLIC = 0.6
ROAD_CAPACITY = 10000

model1 = TransportModel()


CongestionPlot = make_plot_component("congestion_level")
GHGPlot = make_plot_component("total_ghg_sum")
ProfitPlot = make_plot_component("total_system_profit")
ModePlot = make_plot_component({
    "car_share_pct": (1.0, 0.0, 0.0),        # Red
    "bus_share_pct": (0.0, 0.0, 1.0),         # Blue
    "train_share_pct": (0.0, 0.5, 0.0),       # Darker Green
    "bike_walk_share_pct": (0.0, 0.0, 0.0),   # White
})
UpperModePlot = make_plot_component({
    "car_share_pct_upper": (1.0, 0.0, 0.0),        # Red
    "bus_share_pct_upper": (0.0, 0.0, 1.0),         # Blue
    "train_share_pct_upper": (0.0, 0.5, 0.0),       # Darker Green
    "bike_walk_share_pct_upper": (0.0, 0.0, 0.0),   # White
})
MiddleModePlot = make_plot_component({
    "car_share_pct_middle": (1.0, 0.0, 0.0),        # Red
    "bus_share_pct_middle": (0.0, 0.0, 1.0),         # Blue
    "train_share_pct_middle": (0.0, 0.5, 0.0),       # Darker Green
    "bike_walk_share_pct_middle": (0.0, 0.0, 0.0),   # White
})
LowerModePlot = make_plot_component({
    "car_share_pct_lower": (1.0, 0.0, 0.0),        # Red
    "bus_share_pct_lower": (0.0, 0.0, 1.0),         # Blue
    "train_share_pct_lower": (0.0, 0.5, 0.0),       # Darker Green
    "bike_walk_share_pct_lower": (0.0, 0.0, 0.0),   # White
})

model_params = {
    "num_agents": NUM_AGENTS,
    "fare_discount": Slider("Public Transport Fare Discount (%)", 0.0, 0.0, 1.0, 0.005),
    "car_toll": Slider("Car Toll ($)", 0.0, 0.0, 20.0, 0.5),
    "car_cost": CAR_COST,
    "bus_cost": BUS_COST,
    "train_cost": TRAIN_COST,
    "lambda_private": LAMBDA_PRIVATE,
    "lambda_public": LAMBDA_PUBLIC,
    "road_capacity": ROAD_CAPACITY,
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
}
model_params_batch = {
    "num_agents": NUM_AGENTS,
    "fare_discount": [0.0, 0.25, 0.5, 0.75, 1.0],
    "car_toll": range(0,20,5),
    "car_cost": CAR_COST,
    "bus_cost": BUS_COST,
    "train_cost": TRAIN_COST,
    "lambda_private": LAMBDA_PRIVATE,
    "lambda_public": LAMBDA_PUBLIC,
    "road_capacity": ROAD_CAPACITY,
}

def PercentageDisplay(model):
    if model is None:
        return

    with solara.Card(title="Transportation Mode Shares"):
        solara.Text(f"🚗 Car Share: {model.car_share_pct:.1f}%")
        solara.Text(f"🚌 Bus Share: {model.bus_share_pct:.1f}%")
        solara.Text(f"🚆 Train Share: {model.train_share_pct:.1f}%")
        solara.Text(f"🚶‍♂️ Bike/Walk Share: {model.bike_walk_share_pct:.1f}%")
def PercentageDisplayUpper(model):
    if model is None:
        return

    with solara.Card(title="Upper Class Transportation Mode Shares"):
        solara.Text(f"🚗 Car Share: {model.car_share_pct_upper:.1f}%")
        solara.Text(f"🚌 Bus Share: {model.bus_share_pct_upper:.1f}%")
        solara.Text(f"🚆 Train Share: {model.train_share_pct_upper:.1f}%")
        solara.Text(f"🚶‍♂️ Bike/Walk Share: {model.bike_walk_share_pct_upper:.1f}%")
def PercentageDisplayMiddle(model):
    if model is None:
        return

    with solara.Card(title="Middle Class Transportation Mode Shares"):
        solara.Text(f"🚗 Car Share: {model.car_share_pct_middle:.1f}%")
        solara.Text(f"🚌 Bus Share: {model.bus_share_pct_middle:.1f}%")
        solara.Text(f"🚆 Train Share: {model.train_share_pct_middle:.1f}%")
        solara.Text(f"🚶‍♂️ Bike/Walk Share: {model.bike_walk_share_pct_middle:.1f}%")
def PercentageDisplayLower(model):
    if model is None:
        return

    with solara.Card(title="Lower Class Transportation Mode Shares"):
        solara.Text(f"🚗 Car Share: {model.car_share_pct_lower:.1f}%")
        solara.Text(f"🚌 Bus Share: {model.bus_share_pct_lower:.1f}%")
        solara.Text(f"🚆 Train Share: {model.train_share_pct_lower:.1f}%")
        solara.Text(f"🚶‍♂️ Bike/Walk Share: {model.bike_walk_share_pct_lower:.1f}%")

@solara.component
def GHGPolicyPlot(model):
    results = mesa.batch_run(
        TransportModel,
        parameters=model_params_batch,
        iterations=5,
        max_steps=5,
        number_processes=1,
        data_collection_period=9,
        display_progress=True,
    )


    df = pd.DataFrame(results)
    df["policy_label"] = df.apply(
        lambda row: f"transit discount={row['fare_discount']}, toll={row['car_toll']}", axis=1
    )
    grouped = df.groupby(["policy_label", "Step"]).mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(24, 14), dpi=150)
    for label in grouped["policy_label"].unique():
        sub = grouped[grouped["policy_label"] == label]
        ax.plot(sub["Step"], sub["total_ghg"], label=label)

    ax.set_title("Total GHG Emissions Over Time by Policy Suite")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total GHG Emissions")
    ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid(True)
    fig.tight_layout()

    return solara.FigureMatplotlib(fig)



page = SolaraViz(
    model1,
    components=[ModePlot, PercentageDisplay, 
                UpperModePlot, PercentageDisplayUpper, 
                MiddleModePlot, PercentageDisplayMiddle, 
                LowerModePlot, PercentageDisplayLower, 
                CongestionPlot, GHGPlot, ProfitPlot],
    model_params=model_params,
    name="Transportation Mode Choice Model",
)

page
