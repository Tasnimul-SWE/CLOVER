# CLOVER-A Cross-Cancer Learning Model using Variant Data for Biomarker Recognition

Prostate cancer (PCa) is a prevalent malignancy with variable treatment responses. Identifying genetic biomarkers can enhance
personalized treatment strategies. In this study, we introduce CLOVER (a Cross-cancer Learning model using sOmatic Variant data
for biomarkEr Recognition), a deep learning framework that integrates prostate, breast, ovarian, pancreatic and colorectal cancer
data given their molecular similarities in their DNA repair pathway abnormalities to identify jointly important genetic biomarkers.
We introduce a novel framework that utilizes an autoencoder for dimensionality reduction, with the latent space representation
employed in a supervised deep learning architecture for classification. Our model achieved a balanced accuracy of 87%, significantly
outperforming models using single cancer types. Next, we identified key genetic variants contributing to disease states. These
findings highlight the potential of cross-cancer learning in biomarker discovery and personalized medicine by allowing data enrichment
and multi-cancer treatment strategies. To further investigate the biological significance of our proposed biomarkers, we conducted
pathway enrichment analysis to identify the key signaling pathways most impacted by these biomarkers. Additionally, we validated
our results by constructing a separate deep neural network (DNN), which demonstrated that the proposed biomarkers for prostate
cancer achieved better performance compared to randomly chosen biomarkers. To validate our findings from a mutational pattern
perspective, we performed a mutational signature analysis using a non-negative matrix factorization (NMF) approach. The results of
the mutational signature analysis align with the identified biomarkers, which strengthens the robustness of our cross-cancer biomarker
discovery framework.
