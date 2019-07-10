using Models

using Distributions
using Random

M = Models.DistributionModel{Normal,Int,Float64}
model = rand( M, seed=1 )

N = 10_000
x = zeros(N);
Random.seed!( 1 )
Distributions.rand!( model.d, x );

t = 1:N
u = zeros(N);
Models.rand!( model, t, u, seed=1 );

@assert( x == u )

epsilon = 1e-8
p = Models.getparameters( model )

f = (model,t,u) -> [Models.loglikelihood(model,t,u)]

n = length(f(model, t, u))

df = zeros(n,length(p))
Models.dloglikelihood( model, t, u, view(df,1,:) )

fidi = zeros(1,length(p))
for i = 1:length(p)
    lpdp = zeros(2,n)
    for j in 1:2
        pdp = copy(p)
        pdp[i] += (2*j-3)*epsilon
        Models.setparameters!( model, pdp )
        lpdp[j,:] = f( model, t, u )
    end
    fidi[:,i] = diff(lpdp, dims=1)/(2*epsilon)
end
@assert( all(abs.((fidi .- df)./df).<1e-5 ) )
