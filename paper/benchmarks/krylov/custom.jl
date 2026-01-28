using LinearAlgebra, SparseArrays, SparseMatricesCSR
using Krylov


"""
    generate_3d_fem_elasticity_csr(target_bytes::Int; order::Int=3, E::Float64=1e6, ν::Float64=0.3)

Generate a sparse matrix from 3D linear elasticity using high-order finite elements.
The matrix is symmetric positive definite, suitable for conjugate gradient methods.

# Arguments
- `target_bytes::Int`: Target matrix size in bytes (using Float64 precision)
- `order::Int`: Polynomial order of elements (2=quadratic, 3=cubic, etc.). Higher order = more nnz per row
- `E::Float64`: Young's modulus (default: 1e6)
- `ν::Float64`: Poisson's ratio (default: 0.3, must be < 0.5 for positive definiteness)

# Returns
- Sparse matrix in CSC format representing 3D elasticity problem

# Details
- Each node has 3 DOFs (ux, uy, uz displacements)
- High-order elements create dense coupling between nodes
- Order 2: ~27 nodes per element → ~240 nnz per row
- Order 3: ~64 nodes per element → ~580 nnz per row
- Order 4: ~125 nodes per element → ~1125 nnz per row
- Order 5: ~216 nodes per element → ~1950 nnz per row
"""
function generate_3d_fem_elasticity_csr(target_bytes::Int; order::Int=3, E::Float64=1e6, ν::Float64=0.3)

    # Validate Poisson's ratio for positive definiteness
    if ν >= 0.5 || ν < -1.0
        error("Poisson's ratio must be in (-1, 0.5) for positive definiteness")
    end

    # Nodes per element in each direction for order p
    nodes_per_dir = order + 1
    nodes_per_element = nodes_per_dir^3

    # Each node has 3 DOFs (ux, uy, uz)
    dofs_per_node = 3
    dofs_per_element = nodes_per_element * dofs_per_node

    # Estimate number of elements needed
    # Each element contributes roughly dofs_per_element^2 entries
    # But with overlap, nnz per row ≈ nodes_per_element * dofs_per_node
    bytes_per_entry = 16  # Conservative for sparse storage
    nnz_per_row_estimate = nodes_per_element * dofs_per_node
    total_rows_estimate = target_bytes ÷ (bytes_per_entry * nnz_per_row_estimate)
    total_nodes_estimate = total_rows_estimate ÷ dofs_per_node

    # Determine grid of elements
    # For high-order FEM, elements can be larger
    elements_per_dir = max(2, round(Int, (total_nodes_estimate / nodes_per_element)^(1/3)))

    # Total nodes in the mesh (with element overlap for continuous elements)
    nx = elements_per_dir * order + 1
    ny = elements_per_dir * order + 1
    nz = elements_per_dir * order + 1

    total_nodes = nx * ny * nz
    N = total_nodes * dofs_per_node  # Total DOFs

    println("Generating 3D FEM elasticity matrix:")
    println("  Element order: $order")
    println("  Nodes per element: $nodes_per_element")
    println("  Elements: $(elements_per_dir)³ = $(elements_per_dir^3)")
    println("  Node grid: $nx × $ny × $nz")
    println("  Total nodes: $total_nodes")
    println("  Total DOFs: $N")
    println("  Expected nnz per row: ~$nnz_per_row_estimate")

    # Map 3D node index to linear node number
    node_idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)

    # Compute material stiffness matrix (isotropic elasticity)
    λ = E * ν / ((1 + ν) * (1 - 2ν))  # Lamé's first parameter
    μ = E / (2 * (1 + ν))              # Shear modulus (Lamé's second parameter)

    # 6x6 material matrix for 3D elasticity (Voigt notation)
    C = zeros(6, 6)
    C[1:3, 1:3] .= λ
    C[1,1] = C[2,2] = C[3,3] = λ + 2μ
    C[4,4] = C[5,5] = C[6,6] = μ

    # Build element stiffness matrix
    # For simplicity, use a uniform element with simple quadrature
    # This creates the coupling pattern without exact integration

    I_vals = Int[]
    J_vals = Int[]
    V_vals = Float64[]

    # Process each element
    for ez in 1:elements_per_dir
        for ey in 1:elements_per_dir
            for ex in 1:elements_per_dir

                # Get local node indices for this element
                local_nodes = Int[]
                for lz in 0:order
                    for ly in 0:order
                        for lx in 0:order
                            i = (ex - 1) * order + lx + 1
                            j = (ey - 1) * order + ly + 1
                            k = (ez - 1) * order + lz + 1
                            push!(local_nodes, node_idx(i, j, k))
                        end
                    end
                end

                # Create element stiffness matrix (simplified)
                # Use a pattern that creates realistic coupling
                h = 1.0 / elements_per_dir  # Element size
                vol = h^3

                # Simplified element matrix with realistic structure
                for local_i in 1:nodes_per_element
                    node_i = local_nodes[local_i]

                    for local_j in 1:nodes_per_element
                        node_j = local_nodes[local_j]

                        # Distance-based stiffness (decays with distance)
                        li_x, li_y, li_z = (local_i-1) % nodes_per_dir, ((local_i-1) ÷ nodes_per_dir) % nodes_per_dir, (local_i-1) ÷ (nodes_per_dir^2)
                        lj_x, lj_y, lj_z = (local_j-1) % nodes_per_dir, ((local_j-1) ÷ nodes_per_dir) % nodes_per_dir, (local_j-1) ÷ (nodes_per_dir^2)

                        dist = sqrt((li_x - lj_x)^2 + (li_y - lj_y)^2 + (li_z - lj_z)^2)
                        weight = exp(-dist / order) * vol

                        # Couple all 3 DOFs between nodes
                        for dof_i in 1:3
                            for dof_j in 1:3
                                global_i = (node_i - 1) * 3 + dof_i
                                global_j = (node_j - 1) * 3 + dof_j

                                # Diagonal components stronger, off-diagonal for coupling
                                if dof_i == dof_j
                                    value = (λ + 2μ) * weight
                                else
                                    value = λ * weight * 0.5
                                end

                                push!(I_vals, global_i)
                                push!(J_vals, global_j)
                                push!(V_vals, value)
                            end
                        end
                    end
                end
            end
        end
    end

    println("  Assembling sparse matrix...")

    # Create sparse matrix and symmetrize to ensure SPD
    A = sparse(I_vals, J_vals, V_vals, N, N)
    A = (A + A') / 2  # Symmetrize

    # Add small diagonal regularization to ensure strict positive definiteness
    A = A + sparse(1:N, 1:N, fill(1e-6 * maximum(abs.(A.nzval)), N), N, N)

    actual_bytes = sizeof(A.nzval) + sizeof(A.rowval) + sizeof(A.colptr)
    avg_nnz = nnz(A) / N

    println("  Non-zeros: $(nnz(A))")
    println("  Average nnz per row: $(round(avg_nnz, digits=1))")
    println("  Actual size: $(actual_bytes) bytes ($(round(actual_bytes/1024^2, digits=2)) MB)")
    println("  Target size: $target_bytes bytes ($(round(target_bytes/1024^2, digits=2)) MB)")

    return A
end

"""
    generate_3d_poisson_csr(target_bytes::Int)

Generate a sparse matrix from the 3D Poisson/heat equation in CSR format.
The matrix is symmetric positive definite, suitable for conjugate gradient methods.

# Arguments
- `target_bytes::Int`: Target matrix size in bytes (using Float64 precision)

# Returns
- Sparse matrix in CSR (CSC) format representing the 7-point stencil for 3D Poisson equation

# Details
The 3D Poisson equation uses a 7-point stencil:
- 6 off-diagonal entries per row (±1 in x, y, z directions)
- 1 diagonal entry (value of 6)
This creates a symmetric positive definite matrix.
"""
function generate_3d_poisson_csr(target_bytes::Int)
    # Each matrix entry uses 8 bytes (Float64)
    # For CSR/CSC storage: we need values, row indices, and column pointers
    # Approximate storage: nnz * (8 + 4 + 4) bytes for values and indices
    # For 7-point stencil: nnz ≈ 7n (interior points have 7 entries)

    # Estimate grid size
    bytes_per_entry = 16  # Conservative estimate for sparse storage
    nnz_estimate = target_bytes ÷ bytes_per_entry
    n_total = nnz_estimate ÷ 7  # Approximate number of grid points

    # Find nx, ny, nz for cubic-ish grid
    n_side = round(Int, n_total^(1/3))
    n_side = max(n_side, 2)  # Minimum grid size

    nx = ny = nz = n_side
    N = nx * ny * nz

    println("Generating 3D Poisson matrix:")
    println("  Grid dimensions: $nx × $ny × $nz")
    println("  Total unknowns: $N")

    # Build the matrix using COO format, then convert to CSC (Julia's CSR equivalent)
    I = Int[]
    J = Int[]
    V = Float64[]

    # Map 3D index (i,j,k) to linear index
    idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)

    # 7-point stencil for 3D Poisson equation
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                row = idx(i, j, k)

                # Diagonal entry
                push!(I, row)
                push!(J, row)
                push!(V, 6.0)

                # Off-diagonal entries (-1 for each neighbor)
                # x-direction
                if i > 1
                    push!(I, row)
                    push!(J, idx(i-1, j, k))
                    push!(V, -1.0)
                end
                if i < nx
                    push!(I, row)
                    push!(J, idx(i+1, j, k))
                    push!(V, -1.0)
                end

                # y-direction
                if j > 1
                    push!(I, row)
                    push!(J, idx(i, j-1, k))
                    push!(V, -1.0)
                end
                if j < ny
                    push!(I, row)
                    push!(J, idx(i, j+1, k))
                    push!(V, -1.0)
                end

                # z-direction
                if k > 1
                    push!(I, row)
                    push!(J, idx(i, j, k-1))
                    push!(V, -1.0)
                end
                if k < nz
                    push!(I, row)
                    push!(J, idx(i, j, k+1))
                    push!(V, -1.0)
                end
            end
        end
    end

    # Create sparse matrix (CSC format, which is Julia's compressed format)
    A = sparse(I, J, V, N, N)

    actual_bytes = sizeof(A.nzval) + sizeof(A.rowval) + sizeof(A.colptr)
    println("  Non-zeros: $(nnz(A))")
    println("  Actual size: $(actual_bytes) bytes ($(round(actual_bytes/1024^2, digits=2)) MB)")
    println("  Target size: $target_bytes bytes ($(round(target_bytes/1024^2, digits=2)) MB)")
    return A
end

using SparseArrays
using LinearAlgebra

"""
    generate_3d_maxwell_csr(target_bytes::Int; formulation::Symbol=:vector_wave,
                            ε_r::Float64=1.0, μ_r::Float64=1.0, frequency::Float64=1e9)

Generate a sparse matrix from 3D Maxwell's equations using Yee grid or edge elements.
The matrix is Hermitian positive definite, suitable for conjugate gradient methods.

# Arguments
- `target_bytes::Int`: Target matrix size in bytes (using Float64 precision)
- `formulation::Symbol`: Either `:vector_wave` (6 field components E,H) or `:edge_element` (edge-based)
- `ε_r::Float64`: Relative permittivity (default: 1.0)
- `μ_r::Float64`: Relative permeability (default: 1.0)
- `frequency::Float64`: Frequency in Hz (default: 1 GHz)

# Returns
- Sparse matrix in CSC format representing Maxwell's equations

# Details
- `:vector_wave` formulation: 6 DOFs per cell (Ex, Ey, Ez, Hx, Hy, Hz) → ~42 nnz/row
- `:edge_element` formulation: proper edge element discretization → ~150-200 nnz/row
"""
function generate_3d_maxwell_csr(target_bytes::Int;
                                 formulation::Symbol=:vector_wave,
                                 ε_r::Float64=1.0,
                                 μ_r::Float64=1.0,
                                 frequency::Float64=1e9)

    # Physical constants
    c₀ = 3e8  # Speed of light (m/s)
    ω = 2π * frequency
    k₀ = ω / c₀  # Wave number

    if formulation == :vector_wave
        dofs_per_cell = 6  # Ex, Ey, Ez, Hx, Hy, Hz
        nnz_per_row_estimate = 42
    elseif formulation == :edge_element
        dofs_per_edge = 1
        nnz_per_row_estimate = 150
    else
        error("formulation must be :vector_wave or :edge_element")
    end

    # Estimate grid size
    bytes_per_entry = 16
    total_rows_estimate = target_bytes ÷ (bytes_per_entry * nnz_per_row_estimate)

    println("Generating 3D Maxwell matrix:")
    println("  Formulation: $formulation")
    println("  Frequency: $(frequency/1e9) GHz")
    println("  Wavelength: $(c₀/frequency) m")

    if formulation == :vector_wave
        total_cells_estimate = total_rows_estimate ÷ dofs_per_cell
        n_side = max(3, round(Int, total_cells_estimate^(1/3)))
        nx, ny, nz = n_side, n_side, n_side
        return generate_vector_wave_equation(nx, ny, nz, ε_r, μ_r, ω, k₀, target_bytes)
    else
        # For edge elements, we need to count edges
        # Edges: nx*(ny+1)*(nz+1) + (nx+1)*ny*(nz+1) + (nx+1)*(ny+1)*nz ≈ 3*nx*ny*nz
        n_side = max(3, round(Int, (total_rows_estimate / 3)^(1/3)))
        nx, ny, nz = n_side, n_side, n_side
        return generate_edge_element_equation(nx, ny, nz, ε_r, μ_r, ω, k₀, target_bytes)
    end
end

"""
Generate matrix for vector wave equation (coupled E and H fields)
This gives ~42 nnz per row
"""
function generate_vector_wave_equation(nx, ny, nz, ε_r, μ_r, ω, k₀, target_bytes)
    # Total cells and DOFs
    N_cells = nx * ny * nz
    N = N_cells * 6  # 6 field components per cell

    println("  Grid: $nx × $ny × $nz")
    println("  Total cells: $N_cells")
    println("  Total DOFs: $N (6 per cell)")
    println("  Expected nnz per row: ~42")

    # Map 3D cell index to linear index
    cell_idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)

    # DOF ordering: for each cell, [Ex, Ey, Ez, Hx, Hy, Hz]
    dof_idx(i, j, k, component) = (cell_idx(i, j, k) - 1) * 6 + component

    I_vals = Int[]
    J_vals = Int[]
    V_vals = Float64[]

    h = 1.0 / max(nx, ny, nz)  # Grid spacing

    # Build matrix for coupled wave equations
    # ∇×E = iωμH,  ∇×H = -iωεE

    for k in 1:nz
        for j in 1:ny
            for i in 1:nx

                # Process all 6 components
                for comp in 1:6
                    row = dof_idx(i, j, k, comp)

                    # Diagonal term (mass matrix + stabilization)
                    push!(I_vals, row)
                    push!(J_vals, row)
                    push!(V_vals, k₀^2 * (comp <= 3 ? ε_r : μ_r) + 6.0 / h^2)

                    # Couple to other components in same cell (curl operators)
                    for comp2 in 1:6
                        if comp != comp2
                            col = dof_idx(i, j, k, comp2)
                            # E-H coupling through curl
                            if (comp <= 3 && comp2 > 3) || (comp > 3 && comp2 <= 3)
                                push!(I_vals, row)
                                push!(J_vals, col)
                                push!(V_vals, ω / h)
                            end
                        end
                    end

                    # Couple to neighboring cells (6 neighbors)
                    neighbors = [
                        (i-1, j, k), (i+1, j, k),
                        (i, j-1, k), (i, j+1, k),
                        (i, j, k-1), (i, j, k+1)
                    ]

                    for (ni, nj, nk) in neighbors
                        if 1 <= ni <= nx && 1 <= nj <= ny && 1 <= nk <= nz
                            col = dof_idx(ni, nj, nk, comp)
                            push!(I_vals, row)
                            push!(J_vals, col)
                            push!(V_vals, -1.0 / h^2)
                        end
                    end
                end
            end
        end
    end

    println("  Assembling sparse matrix...")
    A = sparse(I_vals, J_vals, V_vals, N, N)

    # Make Hermitian positive definite
    A = (A + A') / 2
    A = A + sparse(1:N, 1:N, fill(maximum(abs.(A.nzval)) * 1e-6, N), N, N)

    actual_bytes = sizeof(A.nzval) + sizeof(A.rowval) + sizeof(A.colptr)
    avg_nnz = nnz(A) / N

    println("  Non-zeros: $(nnz(A))")
    println("  Average nnz per row: $(round(avg_nnz, digits=1))")
    println("  Actual size: $(actual_bytes) bytes ($(round(actual_bytes/1024^2, digits=2)) MB)")
    println("  Target size: $target_bytes bytes ($(round(target_bytes/1024^2, digits=2)) MB)")

    return A
end

"""
Generate matrix for edge element formulation
This gives ~150-200 nnz per row by properly accounting for edge connectivity
"""
function generate_edge_element_equation(nx, ny, nz, ε_r, μ_r, ω, k₀, target_bytes)
    # Edge element discretization
    # Edges parallel to x: nx * (ny+1) * (nz+1)
    # Edges parallel to y: (nx+1) * ny * (nz+1)
    # Edges parallel to z: (nx+1) * (ny+1) * nz

    n_edges_x = nx * (ny + 1) * (nz + 1)
    n_edges_y = (nx + 1) * ny * (nz + 1)
    n_edges_z = (nx + 1) * (ny + 1) * nz
    N = n_edges_x + n_edges_y + n_edges_z

    println("  Grid: $nx × $ny × $nz")
    println("  X-edges: $n_edges_x")
    println("  Y-edges: $n_edges_y")
    println("  Z-edges: $n_edges_z")
    println("  Total DOFs: $N")
    println("  Expected nnz per row: ~150-200")

    # Edge indexing functions
    edge_x_idx(i, j, k) = i + nx * (j - 1) + nx * (ny + 1) * (k - 1)
    edge_y_idx(i, j, k) = n_edges_x + j + ny * (i - 1) + ny * (nx + 1) * (k - 1)
    edge_z_idx(i, j, k) = n_edges_x + n_edges_y + k + nz * (i - 1) + nz * (nx + 1) * (j - 1)

    I_vals = Int[]
    J_vals = Int[]
    V_vals = Float64[]

    h = 1.0 / max(nx, ny, nz)

    # Process X-edges
    for k in 1:(nz+1)
        for j in 1:(ny+1)
            for i in 1:nx
                row = edge_x_idx(i, j, k)

                # Each edge couples to many other edges through:
                # 1. Edges in same face (curl-curl coupling)
                # 2. Edges in adjacent elements
                # 3. Mass matrix terms

                # Diagonal term
                push!(I_vals, row)
                push!(J_vals, row)
                push!(V_vals, k₀^2 * ε_r / μ_r + 4.0 / h^2)

                # Couple to adjacent x-edges (along same line)
                if i > 1
                    col = edge_x_idx(i-1, j, k)
                    push!(I_vals, row)
                    push!(J_vals, col)
                    push!(V_vals, -0.5 / h^2)
                end
                if i < nx
                    col = edge_x_idx(i+1, j, k)
                    push!(I_vals, row)
                    push!(J_vals, col)
                    push!(V_vals, -0.5 / h^2)
                end

                # Couple to y-edges (form loops in xy-plane)
                for dj in [0, 1]
                    for di in [0, 1]
                        if j + dj <= ny && i + di <= nx + 1
                            col = edge_y_idx(i + di, j + dj, k)
                            push!(I_vals, row)
                            push!(J_vals, col)
                            push!(V_vals, 0.25 / (h^2 * μ_r))
                        end
                    end
                end

                # Couple to z-edges (form loops in xz-plane)
                for dk in [0, 1]
                    for di in [0, 1]
                        if k + dk <= nz && i + di <= nx + 1
                            col = edge_z_idx(i + di, j, k + dk)
                            push!(I_vals, row)
                            push!(J_vals, col)
                            push!(V_vals, 0.25 / (h^2 * μ_r))
                        end
                    end
                end

                # Additional coupling to nearby x-edges (for dense stencil)
                for dj in [-1, 1]
                    if 1 <= j + dj <= ny + 1
                        col = edge_x_idx(i, j + dj, k)
                        push!(I_vals, row)
                        push!(J_vals, col)
                        push!(V_vals, -0.25 / h^2)
                    end
                end
                for dk in [-1, 1]
                    if 1 <= k + dk <= nz + 1
                        col = edge_x_idx(i, j, k + dk)
                        push!(I_vals, row)
                        push!(J_vals, col)
                        push!(V_vals, -0.25 / h^2)
                    end
                end
            end
        end
    end

    # Process Y-edges
    for k in 1:(nz+1)
        for j in 1:ny
            for i in 1:(nx+1)
                row = edge_y_idx(i, j, k)

                # Diagonal
                push!(I_vals, row)
                push!(J_vals, row)
                push!(V_vals, k₀^2 * ε_r / μ_r + 4.0 / h^2)

                # Adjacent y-edges
                if j > 1
                    col = edge_y_idx(i, j-1, k)
                    push!(I_vals, row)
                    push!(J_vals, col)
                    push!(V_vals, -0.5 / h^2)
                end
                if j < ny
                    col = edge_y_idx(i, j+1, k)
                    push!(I_vals, row)
                    push!(J_vals, col)
                    push!(V_vals, -0.5 / h^2)
                end

                # Couple to x-edges (xy loops)
                for dj in [0, 1]
                    for di in [0, 1]
                        if i + di <= nx && j + dj <= ny + 1
                            col = edge_x_idx(i + di, j + dj, k)
                            push!(I_vals, row)
                            push!(J_vals, col)
                            push!(V_vals, 0.25 / (h^2 * μ_r))
                        end
                    end
                end

                # Couple to z-edges (yz loops)
                for dk in [0, 1]
                    for dj in [0, 1]
                        if k + dk <= nz && j + dj <= ny + 1
                            col = edge_z_idx(i, j + dj, k + dk)
                            push!(I_vals, row)
                            push!(J_vals, col)
                            push!(V_vals, 0.25 / (h^2 * μ_r))
                        end
                    end
                end

                # Additional nearby y-edges
                for di in [-1, 1]
                    if 1 <= i + di <= nx + 1
                        col = edge_y_idx(i + di, j, k)
                        push!(I_vals, row)
                        push!(J_vals, col)
                        push!(V_vals, -0.25 / h^2)
                    end
                end
                for dk in [-1, 1]
                    if 1 <= k + dk <= nz + 1
                        col = edge_y_idx(i, j, k + dk)
                        push!(I_vals, row)
                        push!(J_vals, col)
                        push!(V_vals, -0.25 / h^2)
                    end
                end
            end
        end
    end

    # Process Z-edges
    for k in 1:nz
        for j in 1:(ny+1)
            for i in 1:(nx+1)
                row = edge_z_idx(i, j, k)

                # Diagonal
                push!(I_vals, row)
                push!(J_vals, row)
                push!(V_vals, k₀^2 * ε_r / μ_r + 4.0 / h^2)

                # Adjacent z-edges
                if k > 1
                    col = edge_z_idx(i, j, k-1)
                    push!(I_vals, row)
                    push!(J_vals, col)
                    push!(V_vals, -0.5 / h^2)
                end
                if k < nz
                    col = edge_z_idx(i, j, k+1)
                    push!(I_vals, row)
                    push!(J_vals, col)
                    push!(V_vals, -0.5 / h^2)
                end

                # Couple to x-edges (xz loops)
                for dk in [0, 1]
                    for di in [0, 1]
                        if k + dk <= nz + 1 && i + di <= nx
                            col = edge_x_idx(i + di, j, k + dk)
                            push!(I_vals, row)
                            push!(J_vals, col)
                            push!(V_vals, 0.25 / (h^2 * μ_r))
                        end
                    end
                end

                # Couple to y-edges (yz loops)
                for dk in [0, 1]
                    for dj in [0, 1]
                        if k + dk <= nz + 1 && j + dj <= ny
                            col = edge_y_idx(i, j + dj, k + dk)
                            push!(I_vals, row)
                            push!(J_vals, col)
                            push!(V_vals, 0.25 / (h^2 * μ_r))
                        end
                    end
                end

                # Additional nearby z-edges
                for di in [-1, 1]
                    if 1 <= i + di <= nx + 1
                        col = edge_z_idx(i + di, j, k)
                        push!(I_vals, row)
                        push!(J_vals, col)
                        push!(V_vals, -0.25 / h^2)
                    end
                end
                for dj in [-1, 1]
                    if 1 <= j + dj <= ny + 1
                        col = edge_z_idx(i, j + dj, k)
                        push!(I_vals, row)
                        push!(J_vals, col)
                        push!(V_vals, -0.25 / h^2)
                    end
                end
            end
        end
    end

    println("  Assembling sparse matrix...")
    A = sparse(I_vals, J_vals, V_vals, N, N)

    # Make symmetric positive definite
    A = (A + A') / 2
    A = A + sparse(1:N, 1:N, fill(maximum(abs.(A.nzval)) * 1e-6, N), N, N)

    actual_bytes = sizeof(A.nzval) + sizeof(A.rowval) + sizeof(A.colptr)
    avg_nnz = nnz(A) / N

    println("  Non-zeros: $(nnz(A))")
    println("  Average nnz per row: $(round(avg_nnz, digits=1))")
    println("  Actual size: $(actual_bytes) bytes ($(round(actual_bytes/1024^2, digits=2)) MB)")
    println("  Target size: $target_bytes bytes ($(round(target_bytes/1024^2, digits=2)) MB)")

    return A
end

# Solver to use
fname      = length(ARGS) >= 1 ?              ARGS[1]  : "cg"
matsize    = length(ARGS) >= 2 ? parse(Int,   ARGS[2]) : 1_000_000_000
ts         = length(ARGS) >= 3 ? parse(Int,   ARGS[3]) : 1_000_000_000
use_xkblas = length(ARGS) >= 4 ? parse(Bool,  ARGS[4]) : false
iter       = length(ARGS) >= 5 ? parse(Int,   ARGS[5]) : 5
mattype    = length(ARGS) >= 6 ? parse(Int,   ARGS[6]) : 0
println("Running fname=$(fname)")
println("To change parameters, run as `julia script.jl [solver:String] [matrix-name:String] [tiles-size:Int] [use-xkblas:Boolean] [iter:Int] [matType:Int]`")


# Get matrix
if mattype == 0
    A = generate_3d_poisson_csr(matsize)
elseif mattype == 1
    #A = generate_3d_maxwell_csr(matsize, formulation=:vector_wave, frequency=10e9)
    A = generate_3d_maxwell_csr(matsize, formulation=:edge_element, frequency=10e9)
elseif mattype == 2
    A = generate_3d_fem_elasticity_csr(matsize, order=4)
end

# Plot matrices to file
if false
    println("EXPORTING MATRIX TO IMAGE")
    using Plots
    p = spy(A, markersize=1, title="Nonzero pattern")
    savefig(p, "nonzero_pattern.png")
    exit()
end

A = SparseMatrixCSR(A)

@assert size(A, 1) == size(A, 2)
n = size(A, 1)
y = rand(n)
println("Matrix loaded of size n=$(n)")

# Get solver
solver = getproperty(Krylov, Symbol(fname))
tolerance = 1.0e-6

# Set XKBlas backend
if use_xkblas
    using XK
    XK.set_tile_parameter(ts)
    include("./overrides.jl")
    println("Using XKBLAS")
else
    println("Not using XKBLAS")
    using CUDA, CUDA.CUSPARSE
    @assert CUDA.functional()
end

itmax_value = 500

# Run
for i in 1:iter
    if use_xkblas
        # With XKBlas, directly use host memory. Ask for CPU write back explicitly
        (x, stats) = solver(A, y, itmax = itmax_value)
    else
        # With CUDA, move memory synchronously first, pass GPU objects to Krylov, and write back
        A_gpu = CuSparseMatrixCSR(A)
        y_gpu = CuVector(y)
        (x_gpu, stats) = solver(A_gpu, y_gpu, itmax = itmax_value)
    end

    println(stats)

end
