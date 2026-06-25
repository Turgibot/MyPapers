# Video Review: זיו חדד 7.1.26 - Soft Local Completeness in XAI

**Video URL:** https://www.youtube.com/watch?v=g-oZijixNsE  
**Date:** January 7, 2026  
**Presenter:** Ziv Weiss Haddad (The Open University / Tel Aviv University)  
**Paper:** "Soft Local Completeness: Rethinking Completeness in XAI" (ICCV 2025)  
**Co-authors:** Oren Barkan, Yehonatan Elisha, Noam Koenigstein

---

## Overview

This video presentation discusses the SLOC (Soft Local Completeness) method for explainable AI (XAI), addressing limitations in traditional completeness-based attribution methods. The work was presented at ICCV 2025 and introduces a novel approach to generating faithful attribution maps by promoting completeness locally within subregions of the attribution map.

## Key Concepts Discussed

### Problem Statement
[**Video Content Needed:** What specific examples or motivations did the presenter use to introduce the problem?]

- **Global Completeness Limitation:** Traditional completeness requires attributions to sum to the model's response globally, but this can be trivially satisfied through post-hoc normalization without genuine explanatory power.
- **Insufficient Criterion:** Well-regarded explanation methods (e.g., Grad-CAM, Meaningful Perturbation) don't inherently satisfy completeness yet produce faithful explanations, suggesting completeness is neither necessary nor sufficient.

### Proposed Solution: Soft Local Completeness

**Core Innovation:**
- **Local vs. Global:** Differentiates between global completeness (entire attribution map) and local completeness (individual subregions/sub-maps).
- **Completeness Gap:** Introduces a flexible measure quantifying deviation of each sub-map from completeness, defined as the difference between the sum of elements in a sub-map and the model's response to the corresponding input subregions.
- **Soft Optimization:** Rather than enforcing strict binary constraints, SLOC minimizes the completeness gap across diverse sub-maps simultaneously through gradient-based optimization.

**Methodology:**
[**Video Content Needed:** Did the presenter show visual examples, mathematical formulations, or implementation details?]

- SLOC promotes completeness locally within sub-maps in a soft, flexible manner.
- Each sub-map is adjusted to achieve local completeness based on the actual impact of corresponding input subregions on the model's output.
- The method emphasizes or attenuates sub-maps based on their actual contribution to predictions.
- Operates in a true black-box setting, requiring only forward passes (no gradient backpropagation through the model).

### Technical Highlights

[**Video Content Needed:** What specific technical details, algorithms, or architectures were discussed?]

- **Black-Box Operation:** Does not require gradient backpropagation through the model; incorporates model predictions as constant nodes in the computation graph.
- **Computational Efficiency:** Simple, computationally efficient gradient expression relying solely on forward passes.
- **No Surrogate Model:** Unlike LTX, avoids learning an "explainer" function, eliminating the need for aligned training datasets.

## Experimental Results

[**Video Content Needed:** What results, benchmarks, or comparisons were shown?]

According to the paper, SLOC demonstrates:
- State-of-the-art results across multiple benchmarks
- Extensive evaluations on various model architectures
- Superior performance compared to existing methods

[**Video Content Needed:** Were specific quantitative results, visual comparisons, or case studies presented?]

## Discussion Points

[**Video Content Needed:** What questions were asked? What clarifications or elaborations did the presenter provide?]

### Key Insights
- Local completeness as a guiding principle rather than a strict global constraint
- The completeness gap as a quantifiable, flexible criterion
- Faithfulness achieved through local optimization rather than global normalization

### Applications
[**Video Content Needed:** Were specific use cases, domains, or applications discussed?]

### Limitations and Future Work
[**Video Content Needed:** Did the presenter discuss limitations, future directions, or open questions?]

## Personal Notes and Observations

[**Video Content Needed:** Add your own observations about presentation style, clarity, visual aids, audience engagement, etc.]

---

## References

- **Paper:** Haddad, Z. W., Barkan, O., Elisha, Y., & Koenigstein, N. (2025). Soft Local Completeness: Rethinking Completeness in XAI. *IEEE International Conference on Computer Vision (ICCV)*.
- **Code Repository:** https://github.com/xaisloc/sloc

---

**Note:** This review is based on the paper content. Sections marked with [**Video Content Needed:**] require information from the actual video presentation to complete. The video is approximately 1 hour and 1 minute long, in Hebrew, with no available transcripts or captions.
