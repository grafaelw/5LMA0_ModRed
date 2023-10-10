function rhs = heatprop2D_rhs(t,T, dummy, Alpha, L, heatSource)
    if (t<10)
        rhs = Alpha*L*T+heatSource;
    else
        rhs = Alpha*L*T;
    end