JULIA = julia 
JLFLAGS = -p 2

LM_SIMS_SOURCE = src/lm_sims.jl
GGM_SIMS_SOURCE = src/ggm_sims.jl

.PHONY: all instantiate lm_sims

all: instantiate lm_sims ggm_sims

instantiate: 
		$(JULIA) -e "using Pkg; Pkg.instantiate()" 

# ---- SIMULATIONS ----
# This could be more abstracted

# - Linear regression
lm_sims: output/BAS.jld2 output/BMA.jld2 output/OccamsLeaps.jld2 output/OccamsSat.jld2 output/OccamsSatNoWindow.jld2 output/OccamsLeapsNoWindow.jld2

output/BAS.jld2: $(LM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(LM_SIMS_SOURCE) BAS

output/BMA.jld2: $(LM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(LM_SIMS_SOURCE) BMA

output/OccamsLeaps.jld2: $(LM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(LM_SIMS_SOURCE) OccamsLeaps

output/OccamsSat.jld2: $(LM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(LM_SIMS_SOURCE) OccamsSat

output/OccamsSatNoWindow.jld2: $(LM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(LM_SIMS_SOURCE) OccamsSatNoWindow

output/OccamsLeapsNoWindow.jld2: $(LM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(LM_SIMS_SOURCE) OccamsLeapsNoWindow

# - GGM
ggm_sims: output/BGGM.jld2 output/BDgraph.jld2 output/Occams.jld2

output/BGGM.jld2: $(GGM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(GGM_SIMS_SOURCE) BGGM
output/BDgraph.jld2: $(GGM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(GGM_SIMS_SOURCE) BDgraph
output/Occams.jld2: $(GGM_SIMS_SOURCE)
		$(JULIA) $(JLFLAGS) $(GGM_SIMS_SOURCE) Occams


