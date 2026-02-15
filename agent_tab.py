"""
Pricing Agent Tab for CreditAI Gradio App
"""

import gradio as gr
import sys
sys.path.append('./agent')
from pricing_agent import DynamicPricingAgent

# Initialize the pricing agent
pricing_agent = DynamicPricingAgent()

def agent_pricing_usa(credit_score, annual_income, loan_amount, existing_debt, employment_years):
    """Get AI agent pricing recommendation for USA applicant"""
    applicant = {
        'id': f'USA_{len(pricing_agent.decision_history) + 1}',
        'country': 'USA',
        'credit_score': int(credit_score),
        'income': float(annual_income),
        'loan_amount': float(loan_amount),
        'existing_debt': float(existing_debt),
        'employment_years': int(employment_years)
    }
    
    decision = pricing_agent.calculate_rate(applicant)
    
    output = f"""
## 🤖 AI Pricing Agent Decision

### Rate Breakdown
- **Base Rate:** {decision['base_rate']:.2%}
- **Risk Adjustment:** {decision['risk_adjustment']:+.2%}
- **Market Adjustment:** {decision['market_adjustment']:+.2%}
- **Profit Adjustment:** {decision['profit_adjustment']:+.2%}

---

### **Final Recommended Rate: {decision['final_rate']:.2%}**

---

### Financial Metrics
- **Expected Annual Profit:** ${decision['expected_profit']:,.2f}
- **Estimated Default Risk:** {decision['default_probability']:.1%}

### 💡 Agent Reasoning
"""
    
    for reason in decision['reasoning']:
        output += f"\n- {reason}"
    
    stats = pricing_agent.get_statistics()
    
    stats_output = f"""
### 📊 Agent Performance
- **Total Decisions:** {stats['total_decisions']}
- **Avg Rate Adjustment:** {stats['avg_rate_adjustment']:+.2%}
- **Avg Final Rate:** {stats['avg_final_rate']:.2%}
- **Total Expected Profit:** ${stats['total_expected_profit']:,.2f}
"""
    
    return output, stats_output


def agent_pricing_india(credit_score, annual_income, loan_amount, existing_debt, employment_years):
    """Get AI agent pricing recommendation for India applicant"""
    applicant = {
        'id': f'IND_{len(pricing_agent.decision_history) + 1}',
        'country': 'India',
        'credit_score': int(credit_score),
        'income': float(annual_income),
        'loan_amount': float(loan_amount),
        'existing_debt': float(existing_debt),
        'employment_years': int(employment_years)
    }
    
    decision = pricing_agent.calculate_rate(applicant)
    
    output = f"""
## 🤖 AI Pricing Agent Decision

### Rate Breakdown
- **Base Rate:** {decision['base_rate']:.2%}
- **Risk Adjustment:** {decision['risk_adjustment']:+.2%}
- **Market Adjustment:** {decision['market_adjustment']:+.2%}
- **Profit Adjustment:** {decision['profit_adjustment']:+.2%}

---

### **Final Recommended Rate: {decision['final_rate']:.2%}**

---

### Financial Metrics
- **Expected Annual Profit:** ₹{decision['expected_profit']:,.2f}
- **Estimated Default Risk:** {decision['default_probability']:.1%}

### 💡 Agent Reasoning
"""
    
    for reason in decision['reasoning']:
        output += f"\n- {reason}"
    
    stats = pricing_agent.get_statistics()
    
    stats_output = f"""
### 📊 Agent Performance
- **Total Decisions:** {stats['total_decisions']}
- **Avg Rate Adjustment:** {stats['avg_rate_adjustment']:+.2%}
- **Avg Final Rate:** {stats['avg_final_rate']:.2%}
- **Total Expected Profit:** ₹{stats['total_expected_profit']:,.2f}
"""
    
    return output, stats_output


def create_pricing_agent_tab():
    """Create the Pricing Agent tab for Gradio"""
    with gr.Blocks() as pricing_tab:
        gr.Markdown("""
        # 🤖 AI Pricing Agent
        
        Optimizes interest rates by balancing risk, market competition, and profit.
        """)
        
        with gr.Tab("USA Pricing"):
            gr.Markdown("### Get AI-powered rate for USA applicant (FICO)")
            
            with gr.Row():
                with gr.Column():
                    usa_fico = gr.Slider(300, 850, 720, step=1, label="FICO Score")
                    usa_income = gr.Number(85000, label="Annual Income ($)")
                    usa_loan = gr.Number(15000, label="Loan Amount ($)")
                    usa_debt = gr.Number(12000, label="Existing Debt ($)")
                    usa_emp = gr.Slider(0, 40, 5, step=1, label="Years Employed")
                    usa_btn = gr.Button("🚀 Get AI Pricing", variant="primary")
                
                with gr.Column():
                    usa_decision = gr.Markdown()
                    usa_stats = gr.Markdown()
            
            usa_btn.click(agent_pricing_usa, [usa_fico, usa_income, usa_loan, usa_debt, usa_emp], [usa_decision, usa_stats])
        
        with gr.Tab("India Pricing"):
            gr.Markdown("### Get AI-powered rate for India applicant (CIBIL)")
            
            with gr.Row():
                with gr.Column():
                    india_cibil = gr.Slider(300, 900, 750, step=1, label="CIBIL Score")
                    india_income = gr.Number(800000, label="Annual Income (₹)")
                    india_loan = gr.Number(500000, label="Loan Amount (₹)")
                    india_debt = gr.Number(200000, label="Existing Debt (₹)")
                    india_emp = gr.Slider(0, 40, 3, step=1, label="Years Employed")
                    india_btn = gr.Button("🚀 Get AI Pricing", variant="primary")
                
                with gr.Column():
                    india_decision = gr.Markdown()
                    india_stats = gr.Markdown()
            
            india_btn.click(agent_pricing_india, [india_cibil, india_income, india_loan, india_debt, india_emp], [india_decision, india_stats])
    
    return pricing_tab
