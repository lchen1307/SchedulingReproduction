\ Model RMP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  Artificial
Subject To
 capacity_cons: - Artificial + 225 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 650 lam_phase1_2 + 150 lam_phase1_3 <= 625
 capacity_cons: - Artificial <= 625
 capacity_cons: - Artificial <= 625
 capacity_cons: - Artificial + 600 lam_phase1_1 + 600 lam_phase1_2
   + 375 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 800 lam_phase1_1 + 150 lam_phase1_2
   + 650 lam_phase1_3 <= 625
 capacity_cons: - Artificial <= 625
 capacity_cons: - Artificial + 600 lam_phase1_1 + 900 lam_phase1_2
   + 575 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 400 lam_phase1_1 + 400 lam_phase1_2
   + 350 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 400 lam_phase1_1 + 425 lam_phase1_2
   + 400 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 75 lam_phase1_1 + 75 lam_phase1_2
   + 75 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 975 lam_phase1_1 <= 625
 capacity_cons: - Artificial + 50 lam_phase1_1 + 700 lam_phase1_2
   + 200 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 100 lam_phase1_1 + 100 lam_phase1_2
   + 1000 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 300 lam_phase1_1 + 100 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 750 lam_phase1_1 + 100 lam_phase1_2
   + 650 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 100 lam_phase1_1 + 75 lam_phase1_2
   + 100 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 300 lam_phase1_1 + 300 lam_phase1_2
   + 300 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 225 lam_phase1_1 + 1200 lam_phase1_2
   + 1425 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 325 lam_phase1_1 + 325 lam_phase1_2
   + 325 lam_phase1_3 <= 625
 capacity_cons: - Artificial + 900 lam_phase1_1 + 900 lam_phase1_2 <= 625
 convex_cons: lam_phase1_1 + lam_phase1_2 + lam_phase1_3 = 1
Bounds
End
