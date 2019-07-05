
struct DistributionModel{D <: Distribution, U, T} <: AbstractModel{U,T}
    d::D
end

