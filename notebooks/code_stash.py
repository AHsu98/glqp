
#Normal equations Approach
def solve_KKT(x,y,s,H,rx,rp,rc,mu,solver = None):
    #Probably don't actually need to do normal equations here
    #Can apply nesterov-todd scaling and do quasi-definite solve
    G = H + C.T@(C.multiply((y/s)[:,None]))
    rhs = -rx - C.T@((1/s)*(y*rp - rc))
    if solver is None:
        solver = qdldl.Solver(G)
    else:
        solver.update(G)
    dx = solver.solve(rhs)
    Cdx = C@dx
    ds = -rp - Cdx
    dy = (1/s) * (-rc + y*rp + y*(Cdx))
    return dx,ds,dy,solver


#Half-Normal Approach
Ik = diags_array(np.ones(k))        
def solve_KKT(x,y,s,H,rx,rp,rc,mu,solver = None):
    #Nesterov-Todd scaling and do quasi-definite
    # Quasi definite for inequality constraints, 
    # "normal equations" Hessian for GLM part
    w = np.sqrt(y/s)
    wC = C.multiply(w[:,None])
    G = block_array(
        [
            [H,wC.T],
            [wC,-Ik]
        ],format = 'csc'
    )
    rhs = np.hstack([-rx,-w*rp + (w/y) * rc])
    if solver is None:
        solver = qdldl.Solver(G)
    else:
        solver.update(G)
    sol = solver.solve(rhs)
    dx = sol[:n]
    dy = w*sol[n:]
    ds = -rp - C@dx
    return dx,ds,dy,solver
