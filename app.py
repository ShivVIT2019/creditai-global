import gradio as gr
from batch_tab import build_batch_tab
import joblib
import pandas as pd
import numpy as np
from agent_tab import create_pricing_agent_tab

# Load the trained model and preprocessors
model = joblib.load('ml/artifacts/credit_model.joblib')
imputer = joblib.load('ml/artifacts/imputer.joblib')
scaler = joblib.load('ml/artifacts/scaler.joblib')
features = joblib.load('ml/artifacts/feature_names.joblib')

# ============================================================
# USA PREDICTION FUNCTION
# ============================================================
def predict_usa(loan_amount, interest_rate, grade, emp_length, home_ownership,
                annual_income, purpose, dti, credit_inquiries, open_accounts,
                revolving_balance, revolving_util, total_accounts, 
                card_type, credit_history_years, payment_history):
    """Predict credit approval for USA applicants"""
    
    # Map inputs to encoded values
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, 
               '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
               '8 years': 8, '9 years': 9, '10+ years': 10}
    home_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}
    purpose_map = {
        'debt_consolidation': 0, 'credit_card': 1, 'home_improvement': 2,
        'other': 3, 'major_purchase': 4, 'small_business': 5,
        'car': 6, 'medical': 7, 'moving': 8, 'vacation': 9,
        'house': 10, 'wedding': 11, 'renewable_energy': 12, 'educational': 13
    }
    
    # Adjust grade based on payment history and credit history
    grade_value = grade_map.get(grade, 1)
    
    # Payment history affects grade
    if payment_history == "Always on time (No missed payments)":
        grade_adjustment = -0.5  # Better grade
    elif payment_history == "Occasional delays (1-2 late payments)":
        grade_adjustment = 0
    else:  # Frequent delays
        grade_adjustment = 1.0  # Worse grade
    
    # Credit history affects grade
    if credit_history_years >= 7:
        grade_adjustment -= 0.3  # Long history helps
    elif credit_history_years < 2:
        grade_adjustment += 0.5  # Short history hurts
    
    grade_value = max(0, min(6, grade_value + grade_adjustment))
    
    # Card type affects interest rate slightly
    card_rate_adjustment = {
        'Visa': 0,
        'Mastercard': 0,
        'American Express': -0.5,  # Amex gets slightly better rate
        'Discover': 0.2
    }
    interest_rate += card_rate_adjustment.get(card_type, 0)
    
    # Calculate installment
    monthly_rate = interest_rate / 100 / 12
    n_payments = 36
    if monthly_rate > 0:
        installment = loan_amount * monthly_rate * (1 + monthly_rate)**n_payments / ((1 + monthly_rate)**n_payments - 1)
    else:
        installment = loan_amount / n_payments
    
    # Create input dictionary
    input_data = {
        'loan_amnt': loan_amount,
        'int_rate': interest_rate,
        'installment': installment,
        'grade': int(grade_value),
        'emp_length': emp_map.get(emp_length, 5),
        'home_ownership': home_map.get(home_ownership, 0),
        'annual_inc': annual_income,
        'verification_status': 1,
        'purpose': purpose_map.get(purpose, 0),
        'dti': dti,
        'delinq_2yrs': 0 if payment_history == "Always on time (No missed payments)" else 1,
        'inq_last_6mths': credit_inquiries,
        'open_acc': open_accounts,
        'pub_rec': 0,
        'revol_bal': revolving_balance,
        'revol_util': revolving_util,
        'total_acc': total_accounts
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict
    probability = model.predict_proba(input_df)[0, 1]
    
    # Adjust probability based on payment history
    if payment_history == "Always on time (No missed payments)":
        probability *= 0.8  # 20% reduction in risk
    elif payment_history == "Frequent delays (3+ late payments)":
        probability *= 1.3  # 30% increase in risk
    
    probability = min(probability, 1.0)  # Cap at 100%
    
    # Make decision
    if probability < 0.3:
        decision = "✅ APPROVED"
        risk_level = "🟢 LOW RISK"
    elif probability < 0.5:
        decision = "✅ APPROVED"
        risk_level = "🟡 MEDIUM RISK"
    else:
        decision = "❌ REJECTED"
        risk_level = "🔴 HIGH RISK"
    
    # Calculate credit limit if approved
    if "APPROVED" in decision:
        base_limit = 5000
        income_factor = min(annual_income / 50000, 3)
        credit_history_factor = min(credit_history_years / 10, 1.5)
        risk_multiplier = 1 - probability
        
        credit_limit = int(base_limit * income_factor * credit_history_factor * (1 + risk_multiplier))
        credit_limit = min(credit_limit, 30000)  # Cap at $30k
        
        # Card type affects limit
        if card_type == "American Express":
            credit_limit = int(credit_limit * 1.2)  # Amex gets 20% more
        
        limit_text = f"${credit_limit:,}"
    else:
        limit_text = "N/A"
    
    # Calculate interest rate recommendation
    if probability < 0.2:
        recommended_apr = "10-13%"
    elif probability < 0.4:
        recommended_apr = "13-18%"
    else:
        recommended_apr = "18-25%"
    
    # Add recommendation message
    if "APPROVED" in decision:
        if card_type == "American Express":
            message = f"💎 Premium approval! {card_type} card with {limit_text} limit. Excellent payment history recognized!"
        else:
            message = f"✨ Approved for {card_type} card! Credit limit: {limit_text}. {credit_history_years} years of credit history helped."
    else:
        if payment_history == "Frequent delays (3+ late payments)":
            message = "⚠️ Application denied due to payment history. Improve on-time payments and reapply in 6 months."
        else:
            message = "⚠️ Application denied. Consider improving: DTI ratio, credit history length, or reducing credit inquiries."
    
    return (
        decision,
        f"{probability*100:.2f}%",
        risk_level,
        limit_text,
        recommended_apr,
        message
    )

# ============================================================
# INDIA PREDICTION FUNCTION  
# ============================================================
def predict_india(loan_amount_inr, cibil_score, city_tier, annual_income_inr,
                  employment_type, loan_purpose, existing_loans, pan_verified,
                  aadhaar_verified, card_type_india, credit_history_india, 
                  payment_history_india):
    """Predict credit approval for India applicants"""
    
    # Convert to USD
    usd_to_inr = 83
    loan_amount_usd = loan_amount_inr / usd_to_inr
    annual_income_usd = annual_income_inr / usd_to_inr
    
    # Map CIBIL to Grade
    if cibil_score >= 800:
        grade = 'A'
        int_rate = 10.5
    elif cibil_score >= 750:
        grade = 'B'
        int_rate = 12.5
    elif cibil_score >= 700:
        grade = 'C'
        int_rate = 15.0
    elif cibil_score >= 650:
        grade = 'D'
        int_rate = 18.0
    elif cibil_score >= 600:
        grade = 'E'
        int_rate = 22.0
    else:
        grade = 'F'
        int_rate = 25.0
    
    # Adjust for verification
    if not pan_verified:
        int_rate += 2
        cibil_score -= 50
    if not aadhaar_verified:
        int_rate += 1.5
        cibil_score -= 30
    
    # Map credit history
    history_map = {
        'New to credit (< 1 year)': 0.5,
        '1-3 years': 2,
        '3-5 years': 4,
        '5-7 years': 6,
        '7+ years': 10
    }
    credit_history_years = history_map.get(credit_history_india, 2)
    
    # Map city tier
    city_map = {
        'Metro (Mumbai, Delhi, Bangalore)': 'MORTGAGE',
        'Tier-1 (Pune, Hyderabad, Chennai)': 'RENT',
        'Tier-2 (Jaipur, Lucknow, Kochi)': 'RENT',
        'Tier-3 (Smaller cities)': 'OWN'
    }
    home_ownership = city_map.get(city_tier, 'RENT')
    
    # Map employment
    emp_map = {
        'Salaried (MNC)': '10+ years',
        'Salaried (Startup)': '3 years',
        'Self-employed': '5 years',
        'Business Owner': '10+ years',
        'Freelancer': '2 years',
        'Student': '< 1 year'
    }
    emp_length = emp_map.get(employment_type, '5 years')
    
    # Map purpose
    purpose_map = {
        'Personal Loan': 'other',
        'Education Loan': 'educational',
        'Home Loan': 'house',
        'Car Loan': 'car',
        'Business Loan': 'small_business',
        'Credit Card': 'credit_card',
        'Debt Consolidation': 'debt_consolidation'
    }
    purpose = purpose_map.get(loan_purpose, 'other')
    
    # Estimate DTI
    estimated_dti = existing_loans * 5
    
    # Map card type (India)
    card_type_usa = {
        'Visa': 'Visa',
        'Mastercard': 'Mastercard',
        'RuPay': 'Visa',  # Map RuPay to Visa equivalent
        'American Express': 'American Express'
    }.get(card_type_india, 'Visa')
    
    # Call USA prediction
    decision, prob, risk, limit_usd, apr, _ = predict_usa(
        loan_amount=loan_amount_usd,
        interest_rate=int_rate,
        grade=grade,
        emp_length=emp_length,
        home_ownership=home_ownership,
        annual_income=annual_income_usd,
        purpose=purpose,
        dti=estimated_dti,
        credit_inquiries=1,
        open_accounts=5,
        revolving_balance=loan_amount_usd * 0.3,
        revolving_util=30,
        total_accounts=8,
        card_type=card_type_usa,
        credit_history_years=credit_history_years,
        payment_history=payment_history_india
    )
    
    # Convert limit to INR
    if limit_usd != "N/A":
        limit_usd_value = int(limit_usd.replace('$', '').replace(',', ''))
        limit_inr = limit_usd_value * usd_to_inr
        limit_text = f"₹{limit_inr:,.0f}"
    else:
        limit_text = "N/A"
    
    # India-specific message
    if "APPROVED" in decision:
        verification_status = ""
        if pan_verified and aadhaar_verified:
            verification_status = "✅ Full verification complete (PAN + Aadhaar)"
        elif pan_verified:
            verification_status = "⚠️ Complete Aadhaar verification for better rates"
        elif aadhaar_verified:
            verification_status = "⚠️ Complete PAN verification required"
        else:
            verification_status = "⚠️ Both PAN and Aadhaar verification needed"
        
        message = f"✅ Approved for {card_type_india} card! CIBIL: {cibil_score} | {verification_status}"
    else:
        if cibil_score < 650:
            message = f"❌ Denied. CIBIL score ({cibil_score}) below minimum 650. Improve credit score and reapply."
        elif not pan_verified or not aadhaar_verified:
            message = "❌ Denied. Complete PAN and Aadhaar verification required for approval."
        else:
            message = f"❌ Denied. High debt-to-income ratio ({existing_loans} existing loans) or insufficient credit history."
    
    return decision, prob, risk, limit_text, apr, message

# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="CreditAI Global") as app:
    
    gr.Markdown("""
    # 💳 CreditAI Global - Smart Credit Approval System
    ### AI-powered credit assessment for USA and India
    """)
    
    with gr.Tabs():
        # ========================================
        # CREDIT RISK TAB
        # ========================================
        with gr.Tab("💳 Credit Risk Assessment"):
            with gr.Row():
                country = gr.Radio(
                    ["🇺🇸 USA", "🇮🇳 India"],
                    label="Select Country",
                    value="🇺🇸 USA",
                    interactive=True
                )
            
            # USA FORM
            with gr.Group(visible=True) as usa_form:
                gr.Markdown("### 🇺🇸 USA Credit Application")
                
                with gr.Row():
                    with gr.Column():
                        usa_loan_amt = gr.Slider(1000, 40000, value=15000, step=500, 
                                                 label="💰 Loan Amount ($)")
                        usa_int_rate = gr.Slider(5, 30, value=12.5, step=0.5,
                                                label="📊 Interest Rate (%)")
                        usa_annual_inc = gr.Number(value=65000, label="💵 Annual Income ($)")
                        usa_dti = gr.Slider(0, 50, value=18.5, step=0.5,
                                           label="📈 Debt-to-Income Ratio (%)")
                        
                    with gr.Column():
                        usa_grade = gr.Dropdown(['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                                               value='B', label="🎯 Credit Grade")
                        usa_emp_length = gr.Dropdown([
                            '< 1 year', '1 year', '2 years', '3 years', '4 years',
                            '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
                        ], value='5 years', label="💼 Employment Length")
                        usa_home = gr.Dropdown(['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
                                              value='MORTGAGE', label="🏠 Home Ownership")
                        usa_purpose = gr.Dropdown([
                            'debt_consolidation', 'credit_card', 'home_improvement',
                            'other', 'major_purchase', 'small_business', 'car',
                            'medical', 'moving', 'vacation', 'house', 'wedding',
                            'renewable_energy', 'educational'
                        ], value='debt_consolidation', label="🎯 Loan Purpose")
                
                gr.Markdown("### 💳 Card & Credit Information")
                with gr.Row():
                    usa_card_type = gr.Dropdown(
                        ['Visa', 'Mastercard', 'American Express', 'Discover'],
                        value='Visa',
                        label="💳 Card Type"
                    )
                    usa_credit_history = gr.Slider(
                        0, 15, value=5, step=1,
                        label="📅 Credit History (years)"
                    )
                    usa_payment_history = gr.Dropdown([
                        'Always on time (No missed payments)',
                        'Occasional delays (1-2 late payments)',
                        'Frequent delays (3+ late payments)'
                    ], value='Always on time (No missed payments)', 
                    label="💯 Payment History")
                
                gr.Markdown("### 📊 Additional Credit Details")
                with gr.Row():
                    usa_inquiries = gr.Slider(0, 10, value=1, step=1,
                                             label="🔍 Credit Inquiries (last 6 months)")
                    usa_open_acc = gr.Slider(0, 30, value=8, step=1,
                                            label="📂 Open Credit Accounts")
                    usa_revol_bal = gr.Number(value=12000, label="💳 Revolving Balance ($)")
                    usa_revol_util = gr.Slider(0, 100, value=45.5, step=0.5,
                                              label="📊 Revolving Utilization (%)")
                    usa_total_acc = gr.Slider(0, 50, value=15, step=1,
                                             label="📋 Total Credit Accounts")
                
                usa_submit = gr.Button("🚀 Check USA Eligibility", variant="primary", size="lg")
            
            # INDIA FORM
            with gr.Group(visible=False) as india_form:
                gr.Markdown("### 🇮🇳 India Credit Application (HDFC Credila Style)")
                
                with gr.Row():
                    with gr.Column():
                        india_loan_amt = gr.Slider(50000, 5000000, value=500000, step=10000,
                                                  label="💰 Loan Amount (₹)")
                        india_cibil = gr.Slider(300, 900, value=750, step=10,
                                               label="📊 CIBIL Score")
                        india_annual_inc = gr.Number(value=800000, label="💵 Annual Income (₹)")
                        india_existing_loans = gr.Slider(0, 5, value=0, step=1,
                                                        label="📋 Existing Loans")
                        
                    with gr.Column():
                        india_city = gr.Dropdown([
                            'Metro (Mumbai, Delhi, Bangalore)',
                            'Tier-1 (Pune, Hyderabad, Chennai)',
                            'Tier-2 (Jaipur, Lucknow, Kochi)',
                            'Tier-3 (Smaller cities)'
                        ], value='Metro (Mumbai, Delhi, Bangalore)', label="🏙️ City Tier")
                        
                        india_employment = gr.Dropdown([
                            'Salaried (MNC)', 'Salaried (Startup)',
                            'Self-employed', 'Business Owner',
                            'Freelancer', 'Student'
                        ], value='Salaried (MNC)', label="💼 Employment Type")
                        
                        india_purpose = gr.Dropdown([
                            'Personal Loan', 'Education Loan', 'Home Loan',
                            'Car Loan', 'Business Loan', 'Credit Card',
                            'Debt Consolidation'
                        ], value='Personal Loan', label="🎯 Loan Purpose")
                
                gr.Markdown("### 💳 Card & Verification")
                with gr.Row():
                    india_card_type = gr.Dropdown(
                        ['Visa', 'Mastercard', 'RuPay', 'American Express'],
                        value='Visa',
                        label="💳 Card Type"
                    )
                    india_credit_history = gr.Dropdown([
                        'New to credit (< 1 year)',
                        '1-3 years',
                        '3-5 years',
                        '5-7 years',
                        '7+ years'
                    ], value='3-5 years', label="📅 Credit History")
                    
                    india_payment_history = gr.Dropdown([
                        'Always on time (No missed payments)',
                        'Occasional delays (1-2 late payments)',
                        'Frequent delays (3+ late payments)'
                    ], value='Always on time (No missed payments)', 
                    label="💯 Payment History")
                
                with gr.Row():
                    india_pan = gr.Checkbox(value=True, label="✅ PAN Card Verified")
                    india_aadhaar = gr.Checkbox(value=True, label="✅ Aadhaar Verified")
                
                india_submit = gr.Button("🚀 Check India Eligibility", variant="primary", size="lg")
            
            # RESULTS (Shared)
            gr.Markdown("---")
            gr.Markdown("### 📊 Assessment Results")
            
            with gr.Row():
                result_decision = gr.Textbox(label="Decision", scale=2)
                result_probability = gr.Textbox(label="Default Probability", scale=1)
            
            with gr.Row():
                result_risk = gr.Textbox(label="Risk Level", scale=1)
                result_limit = gr.Textbox(label="Credit Limit", scale=1)
                result_apr = gr.Textbox(label="Recommended APR", scale=1)
            
            result_message = gr.Textbox(label="💬 Assessment Details", scale=2)
            
            # LOGIC
            def toggle_country(country):
                if country == "🇺🇸 USA":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)
            
            country.change(
                fn=toggle_country,
                inputs=[country],
                outputs=[usa_form, india_form]
            )
            
            usa_submit.click(
                fn=predict_usa,
                inputs=[
                    usa_loan_amt, usa_int_rate, usa_grade, usa_emp_length, usa_home,
                    usa_annual_inc, usa_purpose, usa_dti, usa_inquiries, usa_open_acc,
                    usa_revol_bal, usa_revol_util, usa_total_acc,
                    usa_card_type, usa_credit_history, usa_payment_history
                ],
                outputs=[result_decision, result_probability, result_risk, 
                        result_limit, result_apr, result_message]
            )
            
            india_submit.click(
                fn=predict_india,
                inputs=[
                    india_loan_amt, india_cibil, india_city, india_annual_inc,
                    india_employment, india_purpose, india_existing_loans, india_pan,
                    india_aadhaar, india_card_type, india_credit_history, india_payment_history
                ],
                outputs=[result_decision, result_probability, result_risk, 
                        result_limit, result_apr, result_message]
            )
        
        # ========================================
        # AI PRICING AGENT TAB
        # ========================================
        with gr.Tab("🤖 AI Pricing Agent"):
            create_pricing_agent_tab()

        # ========================================
        # BATCH UNDERWRITING TAB (MPWR-inspired)
        # ========================================
        with gr.Tab("📦 Batch Underwriting"):
            build_batch_tab()
    
    gr.Markdown("""
    ---
    ### 📝 About CreditAI Global
    
    **🇺🇸 USA Model:** Trained on 2.2M real loans from Lending Club (2007-2018)  
    **🇮🇳 India Model:** Adapted from USA model with Indian credit parameters  
    **🎯 Accuracy:** 86%+ ROC-AUC on test data  
    **📊 Features:** 17+ key financial indicators analyzed  
    **💳 Card Types:** Visa, Mastercard, American Express, Discover (USA) | Visa, Mastercard, RuPay, Amex (India)
    
    **Features:**
    - 💳 Card type selection affects credit limits and rates
    - 📅 Credit history length impacts approval decisions
    - 💯 Payment history significantly affects risk assessment
    - ✅ Aadhaar & PAN verification for India (affects rates)
    - 🤖 **AI Pricing Agent** - Dynamic interest rate optimization with RL
    
    *Powered by Random Forest ML Algorithm + Dynamic Pricing Agent*
    """)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
