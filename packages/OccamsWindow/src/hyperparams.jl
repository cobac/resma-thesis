Base.@kwdef struct OccamsWindowParams{F<:AbstractFloat}
    Oᵣ::F = 0.0 # log(1)
    Oₗ::F = log(20.0)
    startup::Symbol = :saturated
end
