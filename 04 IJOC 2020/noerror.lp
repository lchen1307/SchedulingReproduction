\ Model RDWM
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  Artificial
Subject To
 Capacity_Cons: - Artificial + lam_phase1_0 = 2
 Assignment_Cons: - Artificial + lam_phase1_0 = 1
 Assignment_Cons: - Artificial + lam_phase1_0 = 1
 Assignment_Cons: - Artificial = 1
 Assignment_Cons: - Artificial = 1
 Assignment_Cons: - Artificial + lam_phase1_0 = 1
 convex_cons: lam_phase1_0 = 1
Bounds
End
