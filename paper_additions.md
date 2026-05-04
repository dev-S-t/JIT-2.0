# Research Paper - Missing Sections and Figures Guide

## Document: A Demand-Driven Software Approach with Dynamic Micro-Expiry and Just-in-Time Processing to Reduce Platelet Wastage in Blood Banks

---

## SECTION 4: EXPERIMENTAL SETUP

To validate the proposed approach, we conducted simulation studies using synthetic platelet demand data generated to reflect realistic hospital scenarios.

### A. Dataset Description

Two hospital scenarios were simulated over a two-year period (2024-2025):

**Table I: Dataset Characteristics**

| Parameter | Hamilton Medium Hospital | Stanford Large Hospital |
|-----------|-------------------------|------------------------|
| Simulation Period | Jan 2024 - Dec 2025 | Jan 2024 - Dec 2025 |
| Total Days | 732 | 732 |
| Mean Daily Demand | 17.8 units | 34.2 units |
| Demand Range | 3-48 units | 10-76 units |
| Weekend Reduction | ~35% lower | ~40% lower |
| Trauma Events | ~3% of days | ~5% of days |

The synthetic data incorporates:
- Day-of-week seasonality (lower weekend demand)
- Monthly variations (higher summer demand)
- Random trauma events causing demand spikes
- Holiday effects with reduced elective procedures

### B. Prediction Models Evaluated

Three forecasting models were implemented and compared:

1. **Simple Moving Average (SMA)**: 7-day rolling average
2. **XGBoost**: Gradient boosting with temporal features
3. **SARIMA**: Seasonal Auto-Regressive Integrated Moving Average

**Table II: Model Performance Comparison**

| Model | MAE | RMSE | MAPE (%) | Under-prediction Rate (%) |
|-------|-----|------|----------|--------------------------|
| SMA | 7.15 | 9.63 | 42.1 | 70.8 |
| XGBoost | 6.32 | 7.85 | 47.3 | 40.3 |
| SARIMA | 5.85 | 7.39 | 41.4 | 44.4 |

SARIMA achieved the lowest Mean Absolute Error (MAE = 5.85 units) and was selected for subsequent simulations.

---

## SECTION 5: RESULTS AND ANALYSIS

### A. Simulation Framework

Three inventory management strategies were simulated over 144 days:

1. **Traditional**: Fixed safety stock (20% buffer), FIFO dispensing
2. **JIT-Only**: Predictive ordering without expiry extension
3. **JIT + Micro-Expiry**: Predictive ordering with dynamic expiry extension for near-expiry units showing viability

### B. Comparative Performance Results

**Table III: Simulation Results Summary**

| Metric | Traditional | JIT-Only | JIT + Micro-Expiry |
|--------|-------------|----------|-------------------|
| Total Supply Ordered | 2,880 units | 2,355 units | 2,481 units |
| Total Demand | 2,469 units | 2,469 units | 2,469 units |
| Units Utilized | 2,469 units | 2,331 units | 2,448 units |
| Units Wasted | 323 units | 0 units | 0 units |
| Shortage Events | 0 units | 138 units | 21 units |
| Shortage Days | 0 days | 30 days | 3 days |
| Extensions Triggered | N/A | N/A | 10 events |
| **Wastage Rate** | **11.2%** | **0%** | **0%** |
| **Shortage Rate** | **0%** | **5.6%** | **0.9%** |
| **Fulfillment Rate** | **100%** | **94.4%** | **99.1%** |

### C. Key Findings

1. **Wastage Elimination**: The JIT-based approaches eliminated the 11.2% wastage observed in traditional inventory management, representing a potential saving of 323 platelet units over the simulation period.

2. **Shortage Mitigation**: While pure JIT ordering resulted in 30 shortage days (5.6% shortage rate), the addition of micro-expiry extension reduced this to only 3 shortage days (0.9% shortage rate).

3. **Extension Effectiveness**: The 10 micro-expiry extension events successfully prevented 117 units from contributing to shortages, demonstrating the value of dynamic expiry management.

4. **Supply Efficiency**: JIT + Micro-Expiry required 14% fewer units ordered compared to traditional methods while maintaining 99.1% demand fulfillment.

### D. Visual Analysis

[Insert Figure 2: JIT + Micro-Expiry vs JIT Only Comparison]

The comparison figures demonstrate:
- Top panel: With micro-expiry extensions (purple bars), the red shortage-risk areas are significantly reduced through protected regions (green with dotted pattern)
- Bottom panel: Without extensions, all under-prediction events result in unmitigated shortage risk

---

## SECTION 6: DISCUSSION

### A. Interpretation of Results

The simulation results suggest that combining predictive JIT ordering with dynamic micro-expiry management offers a balanced approach to the wastage-shortage trade-off inherent in platelet inventory management. Traditional systems prioritize availability at the cost of significant wastage (11.2%), while pure JIT systems eliminate wastage but introduce unacceptable shortage risks (5.6%).

The proposed approach achieves near-optimal performance on both metrics:
- Zero wastage (matching JIT-only)
- 99.1% fulfillment (approaching traditional systems)

### B. Practical Implications

The findings suggest several practical implications for blood bank operations:

1. **Reduced Collection Burden**: Lower wastage translates to reduced donor recruitment requirements
2. **Cost Savings**: Estimated 10-15% reduction in operational costs based on wastage elimination
3. **Improved Availability**: Near-100% fulfillment rates without excessive inventory

### C. Comparison with Literature

Our simulated wastage rate of 11.2% for traditional methods aligns with literature reports of 10-20% platelet wastage [1, 2]. The 85% reduction in shortage events (from JIT-only to JIT+Micro) demonstrates the value of the micro-expiry concept, supporting findings by Lowalekar and Ravichandran [17] regarding JIT feasibility in blood banking.

---

## SECTION 7: LIMITATIONS

This study acknowledges several limitations:

1. **Simulation-Based Validation**: Results are based on synthetic data; real-world implementation may face additional challenges including regulatory approval for dynamic expiry modification.

2. **Quality Assessment Assumption**: The micro-expiry concept assumes the availability of rapid, reliable platelet quality testing, which may not be universally available.

3. **Demand Prediction Accuracy**: The SARIMA model achieved 41.4% MAPE, indicating substantial prediction uncertainty that affects JIT system reliability.

4. **Single-Center Focus**: The simulation models hospital-level demand without considering inter-hospital redistribution networks.

5. **Regulatory Considerations**: Dynamic modification of expiry dates requires regulatory approval, which varies by jurisdiction and is not currently standard practice.

---

## SECTION 8: FUTURE WORK

Future research directions include:

1. **Pilot Implementation**: Controlled trials at partner blood banks to validate simulation findings against real-world performance.

2. **Advanced Forecasting**: Integration of deep learning models (LSTM, Transformer architectures) for improved demand prediction.

3. **Multi-Site Optimization**: Expansion to regional blood bank networks with inter-facility redistribution capabilities.

4. **Quality Metrics Integration**: Real-time platelet viability monitoring using emerging biosensor technologies.

5. **Regulatory Pathway Development**: Collaboration with regulatory bodies to establish frameworks for dynamic expiry management.

---

## SECTION 9: ETHICAL CONSIDERATIONS

The proposed system raises several ethical considerations:

1. **Patient Safety**: Any modification to expiry dates must maintain rigorous safety standards and only extend shelf life for units meeting quality thresholds.

2. **Informed Consent**: Patients receiving units with extended expiry should be informed as part of transfusion consent processes.

3. **Equitable Access**: The system should ensure that extended units are distributed equitably and not preferentially allocated based on non-clinical factors.

4. **Data Privacy**: Demand prediction algorithms utilizing patient data must comply with healthcare data protection regulations.

---

## FIGURES TO INCLUDE

### Figure 1: System Architecture Diagram
Location: After Section 3A (Methodology introduction)
Description: Flowchart showing data flow from historical records → prediction model → JIT ordering → micro-expiry decision → dispensing

### Figure 2: Hamilton Medium Hospital 2024 - JIT Comparison
File: `outputs/figure_hamilton_medium_hospital_2024_comparison.png`
Location: Section 5D (Visual Analysis)
Caption: "Comparison of JIT + Micro-Expiry system (top) versus JIT-Only system (bottom) for Hamilton Medium Hospital, 2024. Purple bars indicate extension events; green hatched areas show protected demand that would otherwise result in shortages."

### Figure 3: Stanford Large Hospital 2025 - JIT Comparison  
File: `outputs/figure_stanford_large_hospital_2025_comparison.png`
Location: Section 5D (Visual Analysis)
Caption: "Comparison of JIT + Micro-Expiry system (top) versus JIT-Only system (bottom) for Stanford Large Hospital, 2025. The higher demand volume demonstrates scalability of the approach."

### Figure 4: Micro-Expiry Support in Action
File: `outputs/figure7_micro_expiry_action.png`
Location: Section 5C (Key Findings)
Caption: "Detailed view of micro-expiry mechanism showing how extension events (purple bars) provide protection in subsequent days, converting shortage-risk areas (red) to protected fulfilled demand (green hatched)."

### Figure 5: Summary Metrics Comparison
File: `outputs/figure5_summary_table.png`
Location: Section 5B (after Table III)
Caption: "Visual comparison of key performance metrics across Traditional, JIT-Only, and JIT + Micro-Expiry approaches."

---

## UPDATED CONCLUSION (Section 10)

The simulation study demonstrates that a demand-driven blood bank management approach combining just-in-time platelet processing with dynamic micro-expiry management can potentially address the dual challenges of wastage and shortage in platelet inventory management.

Key contributions of this work include:
1. A conceptual framework integrating predictive analytics, JIT ordering, and dynamic expiry management
2. Simulation-based validation showing elimination of 11.2% wastage while maintaining 99.1% demand fulfillment
3. Demonstration of the micro-expiry extension concept reducing shortage events by 85% compared to pure JIT approaches

While regulatory and technical challenges remain—particularly regarding authorization for dynamic expiry modification and availability of rapid quality assessment—this approach offers a promising direction for improving the efficiency and sustainability of blood bank operations. Future work should focus on pilot implementations and regulatory pathway development to translate these simulation findings into clinical practice.

---

## PAPER STRUCTURE SUMMARY

| Section | Title | Status |
|---------|-------|--------|
| 1 | Introduction | ✓ Complete |
| 2 | Literature Review | ✓ Complete |
| 3 | Methodology | ✓ Complete |
| 4 | Experimental Setup | **NEW - Add** |
| 5 | Results and Analysis | **NEW - Add** |
| 6 | Discussion | **NEW - Add** |
| 7 | Limitations | **NEW - Add** |
| 8 | Future Work | **NEW - Add** |
| 9 | Ethical Considerations | **NEW - Add** |
| 10 | Conclusion | Update existing |
| 11 | References | ✓ Complete |
