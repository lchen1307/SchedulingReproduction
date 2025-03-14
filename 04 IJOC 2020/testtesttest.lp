\ Model RDWM
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  Artificial
Subject To
 Capacity_Cons: - Artificial + lam_phase1_lam_phase_1_0_0
   + lam_phase1_lam_phase_1_0_1 + lam_phase1_lam_phase_1_0_2
   + lam_phase1_lam_phase_1_0_3 + lam_phase1_lam_phase_1_0_4
   + lam_phase1_lam_phase_1_0_5 + lam_phase1_lam_phase_1_0_6 = 2
 Assignment_Cons: - Artificial + lam_phase1_lam_phase_1_0_0
   + lam_phase1_lam_phase_1_0_1 + 2 lam_phase1_lam_phase_1_0_2
   + 2 lam_phase1_lam_phase_1_0_3 + 2 lam_phase1_lam_phase_1_0_4
   + 2 lam_phase1_lam_phase_1_0_5 + 2 lam_phase1_lam_phase_1_0_6 = 1
 Assignment_Cons: - Artificial + lam_phase1_lam_phase_1_0_0
   + lam_phase1_lam_phase_1_0_1 + 2 lam_phase1_lam_phase_1_0_2
   + lam_phase1_lam_phase_1_0_3 + lam_phase1_lam_phase_1_0_4
   + lam_phase1_lam_phase_1_0_5 + lam_phase1_lam_phase_1_0_6 = 1
 Assignment_Cons: - Artificial = 1
 Assignment_Cons: - Artificial = 1
 Assignment_Cons: - Artificial + lam_phase1_lam_phase_1_0_0
   + lam_phase1_lam_phase_1_0_1 + lam_phase1_lam_phase_1_0_3
   + lam_phase1_lam_phase_1_0_4 + lam_phase1_lam_phase_1_0_5
   + lam_phase1_lam_phase_1_0_6 = 1
 Convex_Cons: lam_phase1_lam_phase_1_0_0 + lam_phase1_lam_phase_1_0_1
   + lam_phase1_lam_phase_1_0_2 + lam_phase1_lam_phase_1_0_3
   + lam_phase1_lam_phase_1_0_4 + lam_phase1_lam_phase_1_0_5
   + lam_phase1_lam_phase_1_0_6 = 1
Bounds
End
