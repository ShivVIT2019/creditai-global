"""
Dynamic Pricing Agent for CreditAI Global
==========================================

An autonomous AI agent that optimizes interest rates for credit applications
by balancing risk, profitability, and market competitiveness.

Author: Sivasai Atchyut Akella
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from datetime import datetime


class DynamicPricingAgent:
    """
    AI Agent that determines optimal interest rates for credit applications.
    
    Uses:
    - Risk-based pricing
    - Market competition analysis
    - Reinforcement learning for continuous improvement
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the pricing agent.
        
        Args:
            config: Configuration dictionary with base rates, limits, etc.
        """
        self.config = config or self._default_config()
        
        # Base rates by country
        self.base_rates = {
            'USA': 0.12,      # 12% base for USA
            'India': 0.18     # 18% base for India
        }
        
        # Rate adjustment limits
        self.min_rate = 0.08   # 8% minimum
        self.max_rate = 0.36   # 36% maximum
        
        # Decision history for learning
        self.decision_history = []
        
        # Performance tracking
        self.stats = {
            'total_decisions': 0,
            'avg_adjustment': 0.0,
            'total_profit': 0.0,
            'default_rate': 0.0
        }
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'risk_weight': 0.6,
            'market_weight': 0.3,
            'profit_weight': 0.1,
            'learning_rate': 0.01
        }
    
    def calculate_rate(self, applicant: Dict) -> Dict:
        """
        Calculate optimal interest rate for an applicant.
        
        Args:
            applicant: Dictionary with applicant data
                - country: 'USA' or 'India'
                - credit_score: FICO (USA) or CIBIL (India)
                - income: Annual income
                - loan_amount: Requested loan amount
                - employment_years: Years at current job
                - existing_debt: Current debt obligations
                
        Returns:
            Dictionary with rate decision and reasoning
        """
        # Get base rate for country
        base_rate = self.base_rates.get(applicant['country'], 0.15)
        
        # Calculate risk-based adjustment
        risk_adjustment = self._calculate_risk_adjustment(applicant)
        
        # Calculate market-based adjustment  
        market_adjustment = self._calculate_market_adjustment(applicant)
        
        # Calculate profit optimization adjustment
        profit_adjustment = self._calculate_profit_adjustment(applicant)
        
        # Combine adjustments
        total_adjustment = (
            risk_adjustment * self.config['risk_weight'] +
            market_adjustment * self.config['market_weight'] +
            profit_adjustment * self.config['profit_weight']
        )
        
        # Calculate final rate
        final_rate = base_rate + total_adjustment
        
        # Apply limits
        final_rate = max(self.min_rate, min(self.max_rate, final_rate))
        
        # Calculate expected metrics
        default_prob = self._estimate_default_probability(applicant)
        expected_profit = self._calculate_expected_profit(
            applicant['loan_amount'], 
            final_rate, 
            default_prob
        )
        
        # Create decision record
        decision = {
            'timestamp': datetime.now().isoformat(),
            'applicant_id': applicant.get('id', 'unknown'),
            'country': applicant['country'],
            'credit_score': applicant['credit_score'],
            'loan_amount': applicant['loan_amount'],
            'base_rate': round(base_rate, 4),
            'risk_adjustment': round(risk_adjustment, 4),
            'market_adjustment': round(market_adjustment, 4),
            'profit_adjustment': round(profit_adjustment, 4),
            'total_adjustment': round(total_adjustment, 4),
            'final_rate': round(final_rate, 4),
            'default_probability': round(default_prob, 4),
            'expected_profit': round(expected_profit, 2),
            'reasoning': self._generate_reasoning(
                risk_adjustment, 
                market_adjustment, 
                profit_adjustment,
                applicant
            )
        }
        
        # Log decision
        self._log_decision(decision)
        
        return decision
    
    def _calculate_risk_adjustment(self, applicant: Dict) -> float:
        """
        Calculate rate adjustment based on applicant risk.
        
        Higher risk = higher rate
        """
        credit_score = applicant['credit_score']
        country = applicant['country']
        
        # Normalize credit score (FICO vs CIBIL)
        if country == 'USA':
            # FICO: 300-850
            normalized_score = (credit_score - 300) / (850 - 300)
        else:
            # CIBIL: 300-900
            normalized_score = (credit_score - 300) / (900 - 300)
        
        # Calculate debt-to-income ratio
        dti = applicant.get('existing_debt', 0) / max(applicant['income'], 1)
        
        # Calculate loan-to-income ratio
        lti = applicant['loan_amount'] / max(applicant['income'], 1)
        
        # Risk score (0 = low risk, 1 = high risk)
        risk_score = (
            (1 - normalized_score) * 0.5 +  # Credit score (50% weight)
            min(dti, 1.0) * 0.3 +            # DTI ratio (30% weight)
            min(lti, 1.0) * 0.2              # LTI ratio (20% weight)
        )
        
        # Convert to rate adjustment (-3% to +8%)
        adjustment = -0.03 + (risk_score * 0.11)
        
        return adjustment
    
    def _calculate_market_adjustment(self, applicant: Dict) -> float:
        """
        Calculate adjustment based on market competitiveness.
        
        Good applicants = lower rate to win business
        """
        credit_score = applicant['credit_score']
        country = applicant['country']
        
        # For high-quality applicants, reduce rate to compete
        if country == 'USA':
            if credit_score >= 750:
                return -0.02  # Very competitive
            elif credit_score >= 700:
                return -0.01  # Competitive
            else:
                return 0.0    # Standard
        else:  # India
            if credit_score >= 800:
                return -0.02
            elif credit_score >= 750:
                return -0.01
            else:
                return 0.0
    
    def _calculate_profit_adjustment(self, applicant: Dict) -> float:
        """
        Calculate adjustment to maximize profit.
        
        Larger loans = slight reduction to encourage
        """
        loan_amount = applicant['loan_amount']
        
        # Encourage larger loans with small discount
        if loan_amount >= 25000:
            return -0.005  # 0.5% discount for large loans
        elif loan_amount >= 15000:
            return -0.0025
        else:
            return 0.0
    
    def _estimate_default_probability(self, applicant: Dict) -> float:
        """
        Estimate probability of default (simplified model).
        
        In production, this would use your trained ML model.
        """
        credit_score = applicant['credit_score']
        country = applicant['country']
        
        # Normalize score
        if country == 'USA':
            normalized = (credit_score - 300) / (850 - 300)
        else:
            normalized = (credit_score - 300) / (900 - 300)
        
        # Simple logistic-like probability
        # High score = low default probability
        default_prob = 0.25 * (1 - normalized)
        
        # Adjust for DTI
        dti = applicant.get('existing_debt', 0) / max(applicant['income'], 1)
        default_prob += dti * 0.1
        
        return min(default_prob, 0.99)
    
    def _calculate_expected_profit(
        self, 
        loan_amount: float, 
        rate: float, 
        default_prob: float
    ) -> float:
        """
        Calculate expected profit considering default risk.
        
        Expected Profit = (Interest Income) * (1 - Default Probability) - (Loss if Default)
        """
        # Annual interest income
        interest_income = loan_amount * rate
        
        # Expected loss if default (assume 60% recovery)
        loss_if_default = loan_amount * 0.4
        
        # Expected profit
        expected_profit = (
            interest_income * (1 - default_prob) - 
            loss_if_default * default_prob
        )
        
        return expected_profit
    
    def _generate_reasoning(
        self, 
        risk_adj: float, 
        market_adj: float, 
        profit_adj: float,
        applicant: Dict
    ) -> List[str]:
        """Generate human-readable reasoning for the rate decision."""
        reasons = []
        
        # Risk reasoning
        if risk_adj < -0.01:
            reasons.append("✅ Excellent credit profile - reduced risk premium")
        elif risk_adj > 0.05:
            reasons.append("⚠️ Higher risk profile - increased rate for protection")
        else:
            reasons.append("Standard risk assessment")
        
        # Market reasoning
        if market_adj < 0:
            reasons.append("🎯 Competitive rate to win quality applicant")
        
        # Profit reasoning
        if profit_adj < 0:
            reasons.append("💰 Volume discount for larger loan amount")
        
        # Credit score insight
        score = applicant['credit_score']
        if applicant['country'] == 'USA':
            if score >= 750:
                reasons.append("Excellent FICO score (750+)")
            elif score < 650:
                reasons.append("Below-average FICO score")
        else:
            if score >= 800:
                reasons.append("Excellent CIBIL score (800+)")
            elif score < 700:
                reasons.append("Below-average CIBIL score")
        
        return reasons
    
    def _log_decision(self, decision: Dict):
        """Log decision for learning and analytics."""
        self.decision_history.append(decision)
        self.stats['total_decisions'] += 1
        self.stats['avg_adjustment'] = np.mean([
            d['total_adjustment'] for d in self.decision_history
        ])
        self.stats['total_profit'] += decision['expected_profit']
    
    def get_statistics(self) -> Dict:
        """Get agent performance statistics."""
        if not self.decision_history:
            return self.stats
        
        recent_decisions = self.decision_history[-100:]  # Last 100
        
        return {
            'total_decisions': self.stats['total_decisions'],
            'avg_rate_adjustment': round(np.mean([
                d['total_adjustment'] for d in recent_decisions
            ]), 4),
            'avg_final_rate': round(np.mean([
                d['final_rate'] for d in recent_decisions
            ]), 4),
            'total_expected_profit': round(self.stats['total_profit'], 2),
            'avg_default_probability': round(np.mean([
                d['default_probability'] for d in recent_decisions
            ]), 4)
        }
    
    def learn_from_outcome(self, decision_id: str, actual_outcome: Dict):
        """
        Update agent based on actual loan outcomes.
        
        This is where reinforcement learning happens.
        
        Args:
            decision_id: ID of the original decision
            actual_outcome: Dictionary with 'defaulted' (bool) and 'profit' (float)
        """
        # In a full implementation, this would:
        # 1. Compare predicted vs actual default
        # 2. Adjust model weights
        # 3. Update pricing strategy
        
        # Placeholder for now
        pass
    
    def export_decisions(self, filepath: str):
        """Export decision history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
    
    def load_decisions(self, filepath: str):
        """Load decision history from JSON file."""
        with open(filepath, 'r') as f:
            self.decision_history = json.load(f)


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = DynamicPricingAgent()
    
    # Example applicant (USA)
    usa_applicant = {
        'id': 'USA_001',
        'country': 'USA',
        'credit_score': 720,
        'income': 85000,
        'loan_amount': 15000,
        'employment_years': 5,
        'existing_debt': 12000
    }
    
    # Get rate decision
    decision = agent.calculate_rate(usa_applicant)
    
    print("\n🤖 Pricing Agent Decision")
    print("=" * 50)
    print(f"Base Rate:         {decision['base_rate']:.2%}")
    print(f"Risk Adjustment:   {decision['risk_adjustment']:+.2%}")
    print(f"Market Adjustment: {decision['market_adjustment']:+.2%}")
    print(f"Profit Adjustment: {decision['profit_adjustment']:+.2%}")
    print("-" * 50)
    print(f"Final Rate:        {decision['final_rate']:.2%}")
    print(f"\nExpected Profit:   ${decision['expected_profit']:,.2f}")
    print(f"Default Risk:      {decision['default_probability']:.1%}")
    print(f"\n💡 Reasoning:")
    for reason in decision['reasoning']:
        print(f"  • {reason}")
    
    # Example applicant (India)
    india_applicant = {
        'id': 'IND_001',
        'country': 'India',
        'credit_score': 750,
        'income': 800000,  # INR
        'loan_amount': 500000,  # INR
        'employment_years': 3,
        'existing_debt': 200000
    }
    
    decision2 = agent.calculate_rate(india_applicant)
    
    print("\n\n🤖 Pricing Agent Decision (India)")
    print("=" * 50)
    print(f"Base Rate:         {decision2['base_rate']:.2%}")
    print(f"Final Rate:        {decision2['final_rate']:.2%}")
    print(f"Expected Profit:   ₹{decision2['expected_profit']:,.2f}")
    
    # Get statistics
    print("\n\n📊 Agent Statistics")
    print("=" * 50)
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
