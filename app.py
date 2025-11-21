# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle 
# import numpy as np
# import sys
 
# # Patch custom functions if needed
# def ordinal_encode_func(df): return df
# sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# # Layout settings
# st.set_page_config(page_title="📊 Telecom Churn App", layout="wide")
# sns.set(style='whitegrid')
# plt.rcParams['figure.figsize'] = (8, 5)

# # Load Data
# @st.cache_data
# def load_data():
#     return pd.read_csv('Churn_data.csv')

# # Load model, scaler, and model_columns from the pickle file
# @st.cache_resource
# def load_model():
#     with open('advanced_churn_model.pkl', 'rb') as f:
#         model, scaler, model_columns = pickle.load(f)
#     return model, scaler, model_columns

# # Load everything
# data = load_data()
# model, scaler, model_columns = load_model()

# # Sidebar Navigation
# st.sidebar.title("🔍 Navigation")
# page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# # ================== 🏠 MAIN PAGE: CHURN PREDICTION ==================
# if page == "🏠 Churn Prediction":
#     st.title("🔮 Telecom Churn Prediction")
#     st.markdown("Enter important customer details to predict churn likelihood.")

#     # Use only most relevant features
#     col1, col2 = st.columns(2)






#     with col1:
#         tenure = st.slider('Tenure (months)', 0, 100, 12)
#         monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
#         total = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)
#     with col2:
#         contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
#         payment = st.selectbox('Payment Method', [
#             'Electronic check', 'Mailed check',
#             'Bank transfer (automatic)', 'Credit card (automatic)'
#         ])
#         internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

#     # Build user input
#     input_dict = {
#         'tenure': tenure,
#         'MonthlyCharges': monthly,
#         'TotalCharges': total,
#         f'Contract_{contract}': 1,
#         f'PaymentMethod_{payment}': 1,
#         f'InternetService_{internet}': 1,
#     }

#     # Convert to DataFrame and fill missing model columns
#     user_df = pd.DataFrame([input_dict])
#     for col in model_columns:
#         if col not in user_df.columns:
#             user_df[col] = 0  # fill others with 0
#     user_df = user_df[model_columns]  # ensure correct order

#     # Scale and predict
#     if st.button("🔍 Predict Churn"):
#         try:
#             input_scaled = scaler.transform(user_df)
#             prediction = model.predict(input_scaled)[0]
#             probability = model.predict_proba(input_scaled)[0][1] * 100

#             if prediction == 1:
#                 st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
#             else:
#                 st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")

#             # Show Feature Importance if available
#             if hasattr(model, 'feature_importances_'):
#                 st.subheader("📊 Feature Importance (Top 5)")
#                 feat_df = pd.DataFrame({
#                     'feature': model_columns,
#                     'importance': model.feature_importances_
#                 }).sort_values('importance', ascending=False).head(5)
#                 fig, ax = plt.subplots()
#                 ax.barh(feat_df['feature'], feat_df['importance'], color='#4e79a7')
#                 ax.invert_yaxis()
#                 ax.set_xlabel("Importance")
#                 st.pyplot(fig)

#         except Exception as e:
#             st.error(f"❌ Prediction Error: {str(e)}")

# # ================== 📈 INSIGHTS ==================
# elif page == "📈 Insights & Graphs":
#     st.title("📈 Churn Insights & Visualizations")

#     st.subheader("✅ Churn Distribution")
#     churn_counts = data['Churn'].value_counts()
#     fig, ax = plt.subplots()
#     ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
#     ax.bar_label(ax.containers[0])
#     st.pyplot(fig)

#     st.subheader("📑 Churn by Contract Type")
#     churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
#     fig, ax = plt.subplots()
#     ax.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
#     ax.bar_label(ax.containers[0], fmt='%.1f%%')
#     ax.set_ylabel('Churn Rate (%)')
#     st.pyplot(fig)

#     st.subheader("💳 Churn by Payment Method")
#     churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
#     churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
#     fig, ax = plt.subplots()
#     ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
#     ax.bar_label(ax.containers[0], fmt='%.1f%%')
#     st.pyplot(fig)

#     st.markdown("### 🧠 Key Business Insights")
#     st.markdown("""
#     - Month-to-month contracts show the highest churn.
#     - Electronic checks are most churn-prone.
#     - Short-tenure and high-monthly-charge customers are likely to churn.
#     """)

# # ================== 📄 RAW DATA ==================
# elif page == "📄 Raw Data":
#     st.title("📄 Raw Dataset")
#     st.dataframe(data)
#     st.caption(f"Total Records: {len(data)}")



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import sys
 
# ------------------------------------------------------------------------------
# CONFIGURATION & SETUP
# ------------------------------------------------------------------------------

# Patch custom functions to handle pickle loading issues
def ordinal_encode_func(df): return df
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# Layout settings
st.set_page_config(page_title="📊 Telecom Churn App", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# ------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------

@st.cache_data
def load_data():
    # Ensure Churn_data.csv exists in the same directory
    try:
        return pd.read_csv('Churn_data.csv')
    except FileNotFoundError:
        st.error("File 'Churn_data.csv' not found. Please ensure it is in the app directory.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    # Ensure advanced_churn_model.pkl exists in the same directory
    try:
        with open('advanced_churn_model.pkl', 'rb') as f:
            model, scaler, model_columns = pickle.load(f)
        return model, scaler, model_columns
    except FileNotFoundError:
        st.error("File 'advanced_churn_model.pkl' not found.")
        return None, None, None

# Load everything
data = load_data()
model, scaler, model_columns = load_model()

# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------

st.sidebar.title("🔍 Navigation")
# UPDATED ORDER: Data/Insights is now first, Prediction is second
page = st.sidebar.radio("Go to", ["📊 Data & Insights", "🔮 Churn Prediction"])

# ------------------------------------------------------------------------------
# PAGE 1: RAW DATA + INSIGHTS & GRAPHS
# ------------------------------------------------------------------------------
if page == "📊 Data & Insights":
    st.title("📊 Data Overview & Analytics")
    
    if not data.empty:
        # --- SECTION 1: RAW DATA ---
        st.subheader("📄 Raw Data")
        with st.expander("View Full Dataset", expanded=False):
            st.dataframe(data)
            st.caption(f"Total Records: {len(data)}")

        st.markdown("---")

        # --- SECTION 2: GRAPHS & INSIGHTS ---
        st.subheader("📈 Churn Analysis Dashboard")

        # Layout: Create columns for side-by-side graphs
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ Overall Churn Distribution**")
            if 'Churn' in data.columns:
                churn_counts = data['Churn'].value_counts()
                fig1, ax1 = plt.subplots()
                ax1.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
                ax1.bar_label(ax1.containers[0])
                st.pyplot(fig1)
            else:
                st.warning("Column 'Churn' not found in dataset.")

        with col2:
            st.markdown("**📑 Churn by Contract Type**")
            if 'Contract' in data.columns and 'Churn' in data.columns:
                # Calculate percentages
                churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
                fig2, ax2 = plt.subplots()
                ax2.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
                ax2.bar_label(ax2.containers[0], fmt='%.1f%%')
                ax2.set_ylabel('Churn Rate (%)')
                st.pyplot(fig2)

        # Full width graph for Payment Method
        st.markdown("**💳 Churn by Payment Method**")
        if 'PaymentMethod' in data.columns and 'Churn' in data.columns:
            churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
            churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
            ax3.bar_label(ax3.containers[0], fmt='%.1f%%')
            ax3.set_xlabel('Churn Rate (%)')
            st.pyplot(fig3)

        # Key Insights Text
        st.info("""
        **🧠 Key Business Insights:**
        - **Month-to-month contracts** show the highest churn.
        - **Electronic checks** are the most churn-prone payment method.
        - Customers with **short tenure** and **high monthly charges** are at higher risk.
        """)
    else:
        st.warning("No data loaded. Please check your CSV file.")

# ------------------------------------------------------------------------------
# PAGE 2: CHURN PREDICTION
# ------------------------------------------------------------------------------
elif page == "🔮 Churn Prediction":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter customer details below to predict the likelihood of churn.")

    if model is not None:
        # Input Form Container
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 👤 Service Details")
                tenure = st.slider('Tenure (months)', 0, 100, 12)
                monthly = st.number_input('Monthly Charges ($)', 0.0, 500.0, 70.0)
                total = st.number_input('Total Charges ($)', 0.0, 10000.0, 2500.0)
            
            with col2:
                st.markdown("#### 📝 Contract Info")
                contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
                payment = st.selectbox('Payment Method', [
                    'Electronic check', 'Mailed check',
                    'Bank transfer (automatic)', 'Credit card (automatic)'
                ])
                internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

            submitted = st.form_submit_button("🔍 Predict Churn Status")

        # Prediction Logic
        if submitted:
            # 1. Build user input dictionary
            input_dict = {
                'tenure': tenure,
                'MonthlyCharges': monthly,
                'TotalCharges': total,
                'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
                'Contract_One year': 1 if contract == 'One year' else 0,
                'Contract_Two year': 1 if contract == 'Two year' else 0,
                'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
                'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
                'PaymentMethod_Bank transfer (automatic)': 1 if payment == 'Bank transfer (automatic)' else 0,
                'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
                'InternetService_DSL': 1 if internet == 'DSL' else 0,
                'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
                'InternetService_No': 1 if internet == 'No' else 0
            }

            # 2. Convert to DataFrame and align with model columns
            user_df = pd.DataFrame([input_dict])
            
            # Add missing columns with 0 (for One-Hot Encoded features not selected)
            for col in model_columns:
                if col not in user_df.columns:
                    user_df[col] = 0 
            
            # Reorder columns to match training data
            user_df = user_df[model_columns]

            # 3. Scale and Predict
            try:
                input_scaled = scaler.transform(user_df)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1] * 100

                st.markdown("---")
                st.subheader("Prediction Result")
                
                col_res1, col_res2 = st.columns([2, 1])
                
                with col_res1:
                    if prediction == 1:
                        st.error(f"⚠️ **Likely to Churn**\n\nProbability: **{probability:.1f}%**")
                    else:
                        st.success(f"✅ **Not Likely to Churn**\n\nProbability of staying: **{100 - probability:.1f}%**")

                # 4. Feature Importance Visualization
                with col_res2:
                    if hasattr(model, 'feature_importances_'):
                        st.caption("Top Factors influencing this model:")
                        feat_df = pd.DataFrame({
                            'feature': model_columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False).head(5)
                        
                        fig_imp, ax_imp = plt.subplots(figsize=(4, 3))
                        ax_imp.barh(feat_df['feature'], feat_df['importance'], color='#4e79a7')
                        ax_imp.invert_yaxis()
                        ax_imp.set_xlabel("Importance")
                        plt.tight_layout()
                        st.pyplot(fig_imp)

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
    else:
        st.error("Model not loaded. Cannot perform predictions.")
