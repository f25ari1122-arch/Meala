
import streamlit as st
import pandas as pd
import pickle
import zipfile
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="SmartBite - Weekly Meal Planner",
    page_icon="🍽️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


def load_project_files():
    required_files = [
        "price_model.pkl",
        "meal_recommender.pkl",
        "grocery_database.pkl"
    ]

    missing_files = [
        file for file in required_files
        if not (BASE_DIR / file).exists()
    ]

    zip_path = BASE_DIR / "SmartBite_Trained_Models.zip"

    if missing_files and zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)

    missing_files = [
        file for file in required_files
        if not (BASE_DIR / file).exists()
    ]

    if missing_files:
        st.error(
            "Missing model files: " + ", ".join(missing_files)
        )
        st.info(
            "Upload SmartBite_Trained_Models.zip "
            "or all three trained .pkl files beside app.py."
        )
        st.stop()


@st.cache_resource
def load_models():
    with open(BASE_DIR / "price_model.pkl", "rb") as file:
        price_model = pickle.load(file)

    with open(BASE_DIR / "meal_recommender.pkl", "rb") as file:
        meal_data = pickle.load(file)

    with open(BASE_DIR / "grocery_database.pkl", "rb") as file:
        groceries = pickle.load(file)

    return price_model, meal_data, groceries


load_project_files()
price_model, meal_data, groceries = load_models()

vectorizer = meal_data["vectorizer"]
meal_vectors = meal_data["meal_vectors"]
meals = meal_data["meals"]


def predict_grocery_prices(groceries):
    result = groceries.copy()
    current_month = datetime.now().month
    next_month = current_month % 12 + 1

    result["Predicted_Price"] = price_model.predict(
        pd.DataFrame({
            "Item_Code": result["Item_Code"],
            "Month": [next_month] * len(result)
        })
    )

    result["Predicted_Price"] = (
        result["Predicted_Price"].clip(lower=0).round(2)
    )

    return result


def get_meal_cost(meal, grocery_prices, people):
    total = 0

    for item, quantity in meal["Ingredients"].items():
        item_data = grocery_prices[
            grocery_prices["Item"] == item
        ]

        if not item_data.empty:
            price = float(item_data.iloc[0]["Predicted_Price"])
            total += price * quantity * people

    return total


def create_weekly_plan(budget, people, preference, preferred_foods):
    grocery_prices = predict_grocery_prices(groceries)

    query = (
        preferred_foods.strip()
        if preferred_foods.strip()
        else preference
    )

    query_vector = vectorizer.transform([query])
    similarities = (
        query_vector @ meal_vectors.T
    ).toarray()[0]

    meal_candidates = []

    for index, meal in enumerate(meals):
        if preference != "Any":
            if meal["Preference"] != preference:
                continue

        meal_cost = get_meal_cost(
            meal, grocery_prices, people
        )

        meal_candidates.append({
            "Meal": meal,
            "Similarity": similarities[index],
            "Cost": meal_cost
        })

    if not meal_candidates:
        return None

    meal_candidates.sort(
        key=lambda x: x["Similarity"],
        reverse=True
    )

    categories = ["Breakfast", "Lunch", "Dinner"]
    weekly_plan = []
    meal_counts = {}
    total_cost = 0

    for day in range(1, 8):
        for category in categories:
            candidates = [
                item for item in meal_candidates
                if item["Meal"]["Category"] == category
            ]

            candidates.sort(
                key=lambda x: (
                    (1 - x["Similarity"]) * 100
                    + meal_counts.get(
                        x["Meal"]["Name"], 0
                    ) * 20
                )
            )

            selected = None

            for candidate in candidates:
                meal_name = candidate["Meal"]["Name"]
                cost = candidate["Cost"]

                if total_cost + cost <= budget:
                    selected = candidate
                    break

            if selected is None:
                continue

            meal = selected["Meal"]
            total_cost += selected["Cost"]

            meal_counts[meal["Name"]] = (
                meal_counts.get(meal["Name"], 0) + 1
            )

            weekly_plan.append({
                "Day": f"Day {day}",
                "Category": category,
                "Meal": meal["Name"],
                "Estimated Cost (Rs)": round(
                    selected["Cost"], 2
                ),
                "Preparation Time (Minutes)": meal["Prep_Time"]
            })

    if not weekly_plan:
        return None

    grocery_quantities = {}

    for plan in weekly_plan:
        meal = next(
            item for item in meals
            if item["Name"] == plan["Meal"]
        )

        for item, quantity in meal["Ingredients"].items():
            grocery_quantities[item] = (
                grocery_quantities.get(item, 0)
                + quantity * people
            )

    grocery_list = []

    for item, quantity in grocery_quantities.items():
        item_data = grocery_prices[
            grocery_prices["Item"] == item
        ]

        if item_data.empty:
            continue

        price_data = item_data.iloc[0]
        unit_price = float(price_data["Predicted_Price"])

        grocery_list.append({
            "Grocery Item": item,
            "Quantity": round(quantity, 2),
            "Unit": price_data["Unit"],
            "Price per Unit (Rs)": round(unit_price, 2),
            "Total Price (Rs)": round(
                quantity * unit_price, 2
            )
        })

    grocery_df = pd.DataFrame(grocery_list)
    weekly_df = pd.DataFrame(weekly_plan)

    meal_details = []

    for plan in weekly_plan:
        meal = next(
            item for item in meals
            if item["Name"] == plan["Meal"]
        )

        ingredients = ", ".join(
            f"{item}: {quantity * people}"
            for item, quantity in meal["Ingredients"].items()
        )

        meal_details.append({
            "Day": plan["Day"],
            "Meal": meal["Name"],
            "Category": meal["Category"],
            "Ingredients": ingredients,
            "Estimated Cost (Rs)": plan["Estimated Cost (Rs)"],
            "Preparation Time (Minutes)": meal["Prep_Time"]
        })

    details_df = pd.DataFrame(meal_details)

    return {
        "weekly": weekly_df,
        "groceries": grocery_df,
        "details": details_df,
        "prices": grocery_prices,
        "total_cost": round(total_cost, 2),
        "remaining": round(budget - total_cost, 2)
    }


st.title("🍽️ SmartBite")
st.subheader("Smart Weekly Grocery Budget & Meal Planner")

st.write(
    "Plan your weekly meals and grocery shopping "
    "according to your budget."
)

st.sidebar.header("Your Requirements")

budget = st.sidebar.number_input(
    "Weekly Budget (Rs)",
    min_value=100,
    value=5000,
    step=500
)

people = st.sidebar.number_input(
    "Number of People",
    min_value=1,
    value=1,
    step=1
)

preference = st.sidebar.selectbox(
    "Food Preference",
    ["Any", "Vegetarian", "Non-Vegetarian"]
)

preferred_foods = st.sidebar.text_input(
    "Preferred Foods",
    placeholder="e.g. rice, chicken, eggs"
)

if st.sidebar.button("Generate Weekly Plan", type="primary"):
    result = create_weekly_plan(
        budget,
        people,
        preference,
        preferred_foods
    )

    if result is None:
        st.warning(
            "No meals could fit your budget. "
            "Try increasing your budget."
        )
    else:
        st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]

    st.divider()
    st.header("Your Weekly Plan")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Weekly Budget",
        f"Rs {budget:,.2f}"
    )

    col2.metric(
        "Estimated Meal Cost",
        f"Rs {result['total_cost']:,.2f}"
    )

    col3.metric(
        "Remaining Budget",
        f"Rs {result['remaining']:,.2f}"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Weekly Meal Schedule",
        "Grocery Shopping List",
        "Meal Details",
        "Predicted Prices"
    ])

    with tab1:
        st.subheader("7-Day Meal Schedule")
        st.dataframe(
            result["weekly"],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Weekly Schedule",
            result["weekly"].to_csv(index=False),
            "weekly_meal_schedule.csv",
            "text/csv"
        )

    with tab2:
        st.subheader("Grocery Shopping List")
        st.dataframe(
            result["groceries"],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Grocery List",
            result["groceries"].to_csv(index=False),
            "grocery_shopping_list.csv",
            "text/csv"
        )

    with tab3:
        st.subheader("Meal Ingredients and Preparation")
        st.dataframe(
            result["details"],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Meal Details",
            result["details"].to_csv(index=False),
            "meal_details.csv",
            "text/csv"
        )

    with tab4:
        st.subheader("Predicted Grocery Prices")

        price_display = result["prices"][
            ["Item", "Category", "Unit", "Predicted_Price"]
        ].rename(columns={
            "Predicted_Price": "Predicted Price (Rs)"
        })

        st.dataframe(
            price_display,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Predicted Prices",
            price_display.to_csv(index=False),
            "predicted_grocery_prices.csv",
            "text/csv"
        )

else:
    st.info(
        "Enter your requirements in the sidebar "
        "and click Generate Weekly Plan."
    )
