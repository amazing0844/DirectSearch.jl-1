#=
Contains functions/type prototypes for the mesh.
=#

#= Mesh =#
#Subtypes should contain all variables that define a mesh.
#Should make an effort to follow the naming conventions established
#in existing methods in order to give compatibility between methods
abstract type AbstractMesh end
mutable struct Mesh{T} <: AbstractMesh 
    G::Matrix{T}
    D::Matrix{T}
    Δᵐ::T
    Δᵖ::T

    # Override constructor for different default meshes for 
    # different poll techniques.
    Mesh(N::Int64) = Mesh{Float64}(N)
    function Mesh{T}(N::Int64) where T
        mesh = new()
        mesh.G = Matrix(I,N,N)
        mesh.D = hcat(Matrix(I,N,N),-Matrix(I,N,N))
        mesh.Δᵐ = convert(T, 1)
        mesh.Δᵖ = convert(T, 1)
        return mesh
    end 
end

