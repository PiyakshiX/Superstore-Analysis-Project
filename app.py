import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Superstore Sales Prediction",page_icon="📊",layout="wide")
st.markdown("""
<style>
.main{background:#f4f8fb;}
.big{background:linear-gradient(90deg,#1f77b4,#4facfe);padding:18px;border-radius:12px;color:white;text-align:center;}
.pred{background:#e8fff0;border-left:8px solid #16a34a;padding:20px;border-radius:10px;font-size:28px;font-weight:bold;text-align:center;}
footer{visibility:hidden;}
</style>
""",unsafe_allow_html=True)

@st.cache_resource
def load():
    return joblib.load("sales_model.pkl"), joblib.load("model_columns.pkl")
model,columns=load()

st.sidebar.title("📊 Superstore Project")
st.sidebar.success("Machine Learning Model")

st.sidebar.info("""
**Algorithm**
- Random Forest Regressor

**Target**
- Sales

**Dataset**
- Superstore
""")
st.sidebar.write("Developer: **Piyakshi Saikia**")
st.sidebar.markdown("""
### Features
- EDA
- Random Forest Model
- Sales Prediction
""")

st.markdown('<div class="big"><h1>📊 Superstore Sales Prediction Dashboard</h1><p>Predict Sales using Machine Learning</p></div>',unsafe_allow_html=True)
st.write("")

c1,c2=st.columns(2)
with c1:
    quantity=st.number_input("Quantity",1,20,2)
    discount=st.slider("Discount",0.0,0.8,0.2,0.05)
    year=st.selectbox("Order Year",[2014,2015,2016,2017])
    month=st.selectbox("Order Month",list(range(1,13)))
with c2:
    category=st.selectbox("Category",["Furniture","Office Supplies","Technology"])
    subcategory=st.selectbox("Sub-Category",["Accessories","Appliances","Art","Binders","Bookcases","Chairs","Copiers","Envelopes","Fasteners","Furnishings","Labels","Machines","Paper","Phones","Storage","Supplies","Tables"])
    region=st.selectbox("Region",["Central","East","South","West"])
    segment=st.selectbox("Segment",["Consumer","Corporate","Home Office"])
    ship=st.selectbox("Ship Mode",["First Class","Second Class","Standard Class","Same Day"])

if st.button("🚀 Predict Sales",use_container_width=True):
    sample=pd.DataFrame({
        "Quantity":[quantity],
        "Discount":[discount],
        "Order Year":[year],
        "Order Month":[month],
        "Category":[category],
        "Sub-Category":[subcategory],
        "Region":[region],
        "Segment":[segment],
        "Ship Mode":[ship]
    })
    sample=pd.get_dummies(sample)
    sample=sample.reindex(columns=columns,fill_value=0)
    pred=model.predict(sample)[0]
    st.markdown(f"<div class='pred'>💰 Predicted Sales<br><br>₹ {pred:,.2f}</div>",unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📈 Model Information")

a, b, c = st.columns(3)

a.metric("🤖 Model", "Random Forest")
b.metric("📊 Features", "30")
c.metric("🎯 R² Score", "22.8%")

# Dataset Information
st.markdown("### 📂 Dataset")

d1, d2, d3 = st.columns(3)

d1.metric("Rows", "9,994")
d2.metric("Columns", "21")
d3.metric("Target", "Sales")

st.info(
    "This application predicts sales from order details using a trained Random Forest Regression model."
)

st.markdown("---")

st.markdown(
"""
<center>

<b>Developed by Piyakshi Saikia</b><br>

Built with ❤️ using Python • Streamlit • Scikit-learn • Pandas

</center>
""",
unsafe_allow_html=True
)

st.markdown("### Prediction Summary")

summary = pd.DataFrame({
    "Feature": [
        "Quantity",
        "Discount",
        "Category",
        "Sub-Category",
        "Region",
        "Segment",
        "Ship Mode",
        "Order Month",
        "Order Year"
    ],
    "Value": [
        quantity,
        discount,
        category,
        subcategory,
        region,
        segment,
        ship,
        month,
        year
    ]
})

st.table(summary)