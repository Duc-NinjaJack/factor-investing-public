# PHASE 12: LIQUID ALPHA DISCOVERY - PROJECT CHARTER

**Document Version:** 1.0  
**Date Created:** July 28, 2025  
**Owner:** Duc Nguyen, Principal Quantitative Strategist  
**Status:** Active Research Phase  

---

## **PROJECT MISSION**

**Objective:** Conduct a comprehensive factor DNA analysis on the newly defined "ASC-VN-Liquid-150" universe to establish a realistic performance baseline for our existing Quality, Value, and Momentum factors within the investable universe.

**Strategic Context:** This phase represents a critical pivot from our previous "liquidity-last" architecture to a new "liquid-universe-first" methodology. The original strategy showed phenomenal alpha (~2.1 Sharpe) but was concentrated in untradable micro-cap stocks. Phase 12 will determine whether our existing factors retain efficacy in the investable universe.

---

## **RESEARCH QUESTIONS**

### **Primary Research Question**
> "Do our existing Quality, Value, and Momentum factors exhibit meaningful alpha signals within the ASC-VN-Liquid-150 universe?"

### **Specific Sub-Questions**
1. **Factor Dispersion:** Do our factors show sufficient cross-sectional variation in the liquid universe?
2. **Factor Stability:** Are factor signals stable over time within the liquid universe?
3. **Quintile Efficacy:** Do top and bottom quintiles show meaningful performance differences?
4. **Universe Coverage:** Can we achieve adequate factor coverage (125+ stocks) consistently?

---

## **UNIVERSE DEFINITION: ASC-VN-LIQUID-150**

### **Technical Specifications**
- **Selection Criteria:** Top 200 stocks by 63-day Average Daily Trading Value (ADTV)
- **Minimum Threshold:** 10 Billion VND ADTV
- **Refresh Frequency:** Quarterly (Q1, Q2, Q3, Q4)
- **Coverage Requirement:** Minimum 80% trading day coverage during lookback period
- **Look-ahead Prevention:** Universe constructed using T-2 data only

### **Expected Characteristics**
- **Actual Universe Size:** ~150 stocks (hence ASC-VN-Liquid-150)
- **Sector Concentration:** Banking, Real Estate, SOE dominance expected
- **Market Cap Bias:** Large and mid-cap focused
- **Liquidity Range:** 10B+ to 100B+ VND daily turnover

---

## **SUCCESS/FAILURE CRITERIA**

### **Mandatory Sanity Check Gates**
All three checks MUST pass to proceed with analysis:

1. **Coverage Check**
   - **Threshold:** ≥125 stocks with complete factor data
   - **Rationale:** Minimum statistical significance for quintile analysis

2. **Liquidity Overlap Check** 
   - **Threshold:** ≥80% overlap between liquid universe and factor universe
   - **Rationale:** Ensure factor data availability for investable stocks

3. **Factor Dispersion Check**
   - **Threshold:** Cross-sectional standard deviation ≥0.10 for each factor
   - **Rationale:** Minimum signal strength for meaningful ranking

### **Go/No-Go Decision Framework**

**✅ GO Decision (Proceed with Current Factors):**
- All sanity checks pass
- At least 2 of 3 factors show "Strong" efficacy (quintile spread >0.5)
- OR at least 2 of 3 factors show "Moderate+" efficacy (quintile spread >0.2)

**🟡 CAUTIOUS GO (Proceed with Factor Enhancement):**
- All sanity checks pass
- 1-2 factors show "Moderate" efficacy
- Consider factor engineering improvements

**❌ NO-GO (Pivot to New Factor Discovery):**
- Any sanity check fails
- All factors show "Weak" or "Very Weak" efficacy
- Immediate pivot to Liquid Alpha Discovery phase for new factor engineering

---

## **DELIVERABLES & TIMELINE**

### **Week 1: Foundation (July 28 - August 3)**
- [ ] Universe constructor module (`production/universe/`)
- [ ] Sanity check implementation and validation
- [ ] Factor DNA analysis framework

### **Week 2: Analysis (August 4 - August 10)**
- [ ] Complete factor DNA analysis for Q, V, M factors
- [ ] Quintile analysis and performance baseline
- [ ] Go/No-Go decision documentation

### **Week 3: Documentation (August 11 - August 17)**
- [ ] Comprehensive results documentation
- [ ] Decision rationale and next phase planning
- [ ] Handoff documentation for subsequent phases

### **Key Deliverables**
1. **`13_liquid_universe_factor_dna.ipynb`** - Complete analysis notebook
2. **Factor DNA Report** - Detailed factor characteristics in liquid universe
3. **Go/No-Go Decision Document** - Formal recommendation with evidence
4. **Universe Constructor Module** - Production-ready infrastructure

---

## **RISK MITIGATION**

### **Technical Risks**
- **Risk:** Insufficient factor data coverage in liquid universe
- **Mitigation:** Built-in sanity checks will catch this early

- **Risk:** Factor signals too weak in liquid universe  
- **Mitigation:** Clear No-Go criteria prevent wasted effort on weak signals

### **Strategic Risks**
- **Risk:** Temptation to "tune" factors to improve liquid universe performance
- **Mitigation:** No modifications to factor engine during Phase 12. Any changes require separate phase.

- **Risk:** Scope creep into new factor engineering
- **Mitigation:** Strict adherence to Go/No-Go framework. New factors are Phase 13+ only.

---

## **GOVERNANCE & DECISION AUTHORITY**

### **Decision Points**
1. **Sanity Check Gate:** Proceed with analysis Y/N
2. **Factor Efficacy Assessment:** Strong/Moderate/Weak rating for each factor
3. **Final Go/No-Go Decision:** Formal recommendation for next phase

### **Success Metrics**
- **Primary:** Clear, data-driven Go/No-Go decision with supporting evidence
- **Secondary:** Robust universe construction infrastructure for future use
- **Tertiary:** Comprehensive factor baseline for liquid universe

---

## **PHASE COMPLETION CRITERIA**

Phase 12 is considered complete when:
1. ✅ All sanity checks have been executed and documented
2. ✅ Factor DNA analysis completed for all three factors
3. ✅ Go/No-Go decision made and documented with evidence
4. ✅ Next phase direction clearly defined
5. ✅ All code and analysis artifacts properly archived

**Estimated Completion:** August 17, 2025

---

*This charter serves as the single source of truth for Phase 12 scope, objectives, and success criteria. Any modifications require formal documentation and approval.*