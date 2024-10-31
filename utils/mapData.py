def config_3Ddata(type, num):
    if type=="Bumps":
        if num == 3:
            y_i = 39.5 
            y_f = 68.5
            A = 2.5
            B = 12

        if num == 6:
            y_i = 36.5 
            y_f = 71.5
            A = 2.5
            B = 6
        return y_i,y_f,A,B
    
    if type == 'Ridges':
        if num == 4:
            y_i = 39.25
            y_f = 68.75
            A = 2.5
            B = 2.5
            C = 9

        if num == 6:
            y_i = 37.75
            y_f = 70.25
            A = 2.5
            B = 2.5
            C = 6
        return y_i,y_f,A,B,C
    
    if type == 'Waves':
        if num == 4:
            y_i = 38.25
            y_f = 69.75
            A = 2.5
            phi = 0
            # 7 -----> 4 crests and 3 troughs 
            num_amps = 7 
        
        if num == 6:
            y_i = 37.5
            y_f = 70.5
            A = 2.5
            phi = 0
            # 11 -----> 6 crests and 5 troughs 
            num_amps = 11
        return y_i,y_f,A,phi, num_amps
    
    if type == 'Blob':
        if num == 4:
            y_i = 38
            y_f = 70
            A = 2.5
            B_x = 9
            B_y = 9
        
        if num == 6:
            y_i = 36.5
            y_f = 71.5
            A = 2.5
            B_x = 6
            B_y = 6
        return y_i,y_f,A,B_x,B_y   
    

