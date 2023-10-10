function rhs=heatprop2d_rhs(t,Tt,dummy, Alpha_reshaped, KXY, Source_fft)
    
    % Solution with source for the first 10 seconds
    if (t<10)
        rhs = -Alpha_reshaped.*KXY.*Tt + Source_fft;
    else
        rhs =  -Alpha_reshaped*KXY.*Tt;
    end


%     rhs =  -Alpha_reshaped*KXY.*Tt; % Without source solution
    
    