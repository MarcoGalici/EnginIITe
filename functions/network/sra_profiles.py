import os
import numpy as np
import pandas as pd
import xlwings as xw
from tqdm.auto import tqdm


def pl_net1_profiles_load(net, INPUT_PATH, time_step, scenario):
    """
    Function 1: To import the Poland MV Network Profiles
    Demo: Poland
    Versions:
        1st 21/12/2022
        2nd 24/04/2023, after repository update
    """
    # PATH_PL_NET1_PRF = IN_PATH+'/pl_profiles/pl_net1/pl_net1_profiles_inputs.xlsx'
    PATH_PL_NET1_PRF = os.path.join(INPUT_PATH,'profile_files/pl_profiles/pl_net1/pl_net1_profiles_inputs.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_PL_NET1_PRF)
    net1_prf_load = net1_prf_book.sheets[0]
    net1_prf_load_max = net1_prf_load.range('B2:B25').options(pd.DataFrame, index=False,header=False).value
    net1_prf_load_min = net1_prf_load.range('B26:B49').options(pd.DataFrame, index=False,header=False).value
    net1_prf_book.close()
    app.quit()

    # LOAD PROFILES BY LOAD INDEX
    # Empty profiles for P and Q
    P_profile = pd.DataFrame(np.zeros((time_step,len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    Q_profile = pd.DataFrame(np.zeros((time_step,len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    
    # LOAD PROFILES BY BUS INDEX
    # Empty profiles for P and Q
    P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    
    if scenario == "max":
        # net1_prf_load_coeff_max = net1_prf_load_input.iloc[0:24,1]
        # Profiles assigment according to load profile type/number [Mwh]
        for i in net.load.index:
            P_profile.iloc[:,i] = net.load.p_mw[i]*net1_prf_load_max
            Q_profile.iloc[:,i] = net.load.q_mvar[i]*net1_prf_load_max
            
        P_profile1 = P_profile.T
        Q_profile1 = Q_profile.T
        
        for i in net.bus.index:
            for j in net.load.index:
                if net.load.bus[j] == i:
                    P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
                    Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]
    else:
        # net1_prf_load_coeff_min = net1_prf_load_input.iloc[24:48,1]
        # Profiles assigment according to load profile type/number [Mwh]
        for i in net.load.index:
            P_profile.iloc[:,i] = net.load.p_mw[i]*net1_prf_load_min
            Q_profile.iloc[:,i] = net.load.q_mvar[i]*net1_prf_load_min
            
        P_profile1 = P_profile.T
        Q_profile1 = Q_profile.T
        
        for i in net.bus.index:
            for j in net.load.index:
                if net.load.bus[j] == i:
                    P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
                    Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]
    
    return P_profile1, Q_profile1, P_profile2, Q_profile2


def pl_net1_profiles_gen(net, INPUT_PATH, time_step, scenario):
    """
    Function 2: To import Generation Profiles for net1 Poland demo
    Versions:
        1st 21/12/2022
        2nd 24/04/2023, after repository update
    """
    # Import input profiles from German demo data
    # PATH_GE_NET1_PRF = IN_PATH+'/pl_profiles/pl_net1/pl_net1_profiles_inputs.xlsx'
    PATH_PL_NET1_PRF = os.path.join(INPUT_PATH, 'profile_files/pl_profiles/pl_net1/pl_net1_profiles_inputs.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_PL_NET1_PRF)
    # Gen0 coefficients
    net1_prf_gen0 = net1_prf_book.sheets[1]
    net1_prf_gen0_max = net1_prf_gen0.range('B2:B25').options(pd.DataFrame, index=False,header=False).value.astype(float)
    net1_prf_gen0_min = net1_prf_gen0.range('B26:B49').options(pd.DataFrame, index=False,header=False).value.astype(float)
    # Gen1 coefficients
    net1_prf_gen1 = net1_prf_book.sheets[2]
    net1_prf_gen1_max = net1_prf_gen1.range('B2:B25').options(pd.DataFrame, index=False,header=False).value.astype(float)
    net1_prf_gen1_min = net1_prf_gen1.range('B26:B49').options(pd.DataFrame, index=False,header=False).value.astype(float)
    # Gen2 coefficients
    net1_prf_gen2 = net1_prf_book.sheets[3]
    net1_prf_gen2_max = net1_prf_gen2.range('B2:B25').options(pd.DataFrame, index=False,header=False).value.astype(float)
    net1_prf_gen2_min = net1_prf_gen2.range('B26:B49').options(pd.DataFrame, index=False,header=False).value.astype(float) 
    net1_prf_book.close()
    app.quit()

    # GENERATION PROFILES BY GENERATOR INDEX
    # Empty profiles for P and Q considering the network values
    P_profile = pd.DataFrame(np.ones((time_step,len(net.sgen.index))), index=list(range(time_step)), columns=net.sgen.index)*np.nan
    Q_profile = pd.DataFrame(np.ones((time_step,len(net.sgen.index))), index=list(range(time_step)), columns=net.sgen.index)*np.nan
    
    # GENERATION PROFILES BY BUS INDEX
    # Empty profiles for P and Q
    P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))
    Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))
    
    # Multiply the empty profiles by the coefficients, according to the type of generator
    if scenario == "max":
        for i in net.sgen.index:
            if i==0 or i==4: #Wind Swarzewo 1 and 2
                P_profile.iloc[:,i] = net1_prf_gen0_max/1000
                Q_profile.iloc[:,i] = net1_prf_gen0_max*0
            elif i==1: # Wind Lebcz 
                P_profile.iloc[:,i] = net1_prf_gen1_max/1000
                Q_profile.iloc[:,i] = net1_prf_gen1_max*0
            elif i==2: #Wind Polczyno
                P_profile.iloc[:,i] = net1_prf_gen2_max/1000
                Q_profile.iloc[:,i] = net1_prf_gen2_max*0
            elif i==3: #BGP Swarzewo
                P_profile.iloc[:,i] = net.sgen.p_mw[i]*0.6507 #Capacity factor calculated in the input data file of the network
                Q_profile.iloc[:,i] = net.sgen.q_mvar[i]*0.6507 
            elif i==5 or i==6: #CHP Energobaltic 1 and 2
                P_profile.iloc[:,i] = net.sgen.p_mw[i]*0.7887 #Capacity factor calculated in the input data file of the network
                Q_profile.iloc[:,i] = net.sgen.q_mvar[i]*0.7887  
            else:
                P_profile.iloc[:,i] = "nan"
                Q_profile.iloc[:,i] = "nan"
        
        P_profile1 = P_profile.T
        Q_profile1 = Q_profile.T
        
        for i in net.bus.index:
            for j in net.sgen.index:
                if net.sgen.bus[j] == i:
                    P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
                    Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]
    else:
        for i in net.sgen.index:
            if i==0 or i==4: #Wind Swarzewo 1 and 2
                P_profile.iloc[:,i] = net1_prf_gen0_min/1000
                Q_profile.iloc[:,i] = net1_prf_gen0_min*0
            elif i==1: # Wind Lebcz 
                P_profile.iloc[:,i] = net1_prf_gen1_min/1000
                Q_profile.iloc[:,i] = net1_prf_gen1_min*0
            elif i==2: #Wind Polczyno
                P_profile.iloc[:,i] = net1_prf_gen2_min/1000
                Q_profile.iloc[:,i] = net1_prf_gen2_min*0
            elif i==3: #BGP Swarzewo
                P_profile.iloc[:,i] = net.sgen.p_mw[i]*0.6507 # Capacity factor calculated in the input data file of the network
                Q_profile.iloc[:,i] = net.sgen.q_mvar[i]*0.6507 
            elif i==5 or i==6: #CHP Energobaltic 1 and 2
                P_profile.iloc[:,i] = net.sgen.p_mw[i]*0.7887 # Capacity factor calculated in the input data file of the network
                Q_profile.iloc[:,i] = net.sgen.q_mvar[i]*0.7887  
            else:
                P_profile.iloc[:,i] = "nan"
                Q_profile.iloc[:,i] = "nan"
        
        P_profile1 = P_profile.T
        Q_profile1 = Q_profile.T
        
        for i in net.bus.index:
            for j in net.sgen.index:
                if net.sgen.bus[j] == i:
                    P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
                    Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]
    

    return P_profile1, Q_profile1, P_profile2, Q_profile2


def ge_net1_profiles_load(net, INPUT_PATH, time_step, power_factor=0.95):
    """
    Function 3: To import load profiles for German net 1
    Demo: German net 1
    Versions:
        1st 21/12/2022
        2nd 23/04/2023, after repository update
    """
    # Import input profiles from German demo data
    # PATH_GE_NET1_PRF = IN_PATH+'/ge_profiles/ge_net1/ge_net1_profiles_base.xlsx'
    PATH_GE_NET1_PRF = os.path.join(INPUT_PATH, 'profile_files/ge_profiles/ge_net1/ge_net1_profiles_base.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_GE_NET1_PRF)
    net1_prf_sheet = net1_prf_book.sheets[0]
    net1_prf_input = net1_prf_sheet.range('A1').options(pd.DataFrame, index=True,header=True,expand='table').value.astype(float)
    net.load.profile_num = net.load.profile_num.astype(int)
    net1_prf_book.close()
    app.quit()

    # LOAD PROFILES BY LOAD INDEX
    # Empty profiles for P and Q
    P_profile = pd.DataFrame(np.zeros((time_step,len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    Q_profile = pd.DataFrame(np.zeros((time_step,len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    
    # Profiles assigment according to load profile type/number [Mwh]
    for i in tqdm(net.load.index):
        if i==0 or i==1 or i== 2: # MV loads Profiles
            max_coeff=(net1_prf_input.iloc[:, net.load.profile_num[i]]).max()
            max_p= net.load.p_mw[i]/max_coeff
            max_q= net.load.q_mvar[i]/max_coeff
            P_profile.iloc[:,i] = net1_prf_input.iloc[:,net.load.profile_num[i]]*max_p
            Q_profile.iloc[:,i] = net1_prf_input.iloc[:,net.load.profile_num[i]]*max_q
        elif (i>2 and (net.load.profile_type[i] == "Household" or net.load.profile_type[i] == "Commercial" or net.load.profile_type[i] == "Street_lighting")): #Household, Commercial, and Street lighting load Profiles
            P_profile.iloc[:,i] = (net1_prf_input.iloc[:,net.load.profile_num[i]]/1000)*net.load.profile_factor[i] #Only for ge_net1 heat_household points, multiplication of profile factor
            Q_profile.iloc[:,i] = np.sqrt((P_profile.iloc[:,i]/power_factor)**2 - P_profile.iloc[:,i]**2)
        elif(i>2 and (net.load.profile_type[i] == "Heatpump" or net.load.profile_type[i] == "Heatstorage")): #Heatstorage and Heatpump profiles
            P_profile.iloc[:,i] = (net1_prf_input.iloc[:,net.load.profile_num[i]])*(net.load.kwh_heat[i]/1000) # multiply base normalized profile by MWh consumption
            #Q_profile.iloc[:,i] = np.sqrt((P_profile.iloc[:,i]/power_factor)**2 - P_profile.iloc[:,i]**2)
            Q_profile.iloc[:,i] = 0 #Because are electrical heat storage
        else:
            P_profile.iloc[:,i] = "nan"
            Q_profile.iloc[:,i] = "nan"
    
    P_profile1 = P_profile.T
    Q_profile1 = Q_profile.T
    
    #LOAD PROFILES BY BUS INDEX
    # Empty profiles for P and Q
    P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    
    for i in tqdm(net.bus.index):
        for j in net.load.index:
            if net.load.bus[j] == i:
                P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
                Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]

    return P_profile1, Q_profile1, P_profile2, Q_profile2


def ge_net1_profiles_gen(net, INPUT_PATH, time_step):
    """
    Function 4: To import Generation Profiles for net1 German demo
    Demo: German
    Versions:
        1st 21/12/2022
        2nd 23/04/2023, after repository update
    """
    # Import input profiles from German demo data
    # PATH_GE_NET1_PRF = IN_PATH+'/ge_profiles/ge_net1/ge_net1_profiles_base.xlsx'
    PATH_GE_NET1_PRF = os.path.join(INPUT_PATH, 'profile_files/ge_profiles/ge_net1/ge_net1_profiles_base.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_GE_NET1_PRF)
    net1_prf_sheet = net1_prf_book.sheets[0]
    net1_prf_input = net1_prf_sheet.range('A1').options(pd.DataFrame, index=True,header=True,expand='table').value.astype(float)
    net.sgen.profile_num = net.sgen.profile_num.astype(int)
    net1_prf_book.close()
    app.quit()

    # GENERATION PROFILES BY GENERATOR INDEX
    # Empty profiles for P and Q considering the network values
    P_profile = pd.DataFrame(np.ones((time_step,len(net.sgen.index))), index=list(range(time_step)), columns=net.sgen.index)*np.nan
    Q_profile = pd.DataFrame(np.ones((time_step,len(net.sgen.index))), index=list(range(time_step)), columns=net.sgen.index)*np.nan
    
    # Multiply the empty profiles by the coefficients, according to the type of generator
    for i in tqdm(net.sgen.index):
        if net.sgen.profile_type[i] == 'PV':
            P_profile.iloc[:,i] = net.sgen.p_mw[i]*net1_prf_input.iloc[:,net.sgen.profile_num[i]]
            Q_profile.iloc[:,i] = net.sgen.q_mvar[i]*net1_prf_input.iloc[:,net.sgen.profile_num[i]]
        elif net.sgen.profile_type[i] == 'CHP':
            P_profile.iloc[:,i] = net.sgen.p_mw[i]*net1_prf_input.iloc[:,net.sgen.profile_num[i]]*0.75 #it is assumed a capacity factor 0f 75%
            Q_profile.iloc[:,i] = net.sgen.q_mvar[i]*net1_prf_input.iloc[:,net.sgen.profile_num[i]]*0.75
        else:
            P_profile.iloc[:,i] = "nan"
            Q_profile.iloc[:,i] = "nan"

            
    P_profile1 = P_profile.T
    Q_profile1 = Q_profile.T
    
    #GENERATION PROFILES BY BUS INDEX
    # Empty profiles for P and Q
    P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))
    Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))
    
    for i in tqdm(net.bus.index):
        for j in net.sgen.index:
            if net.sgen.bus[j] == i:
                P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
                Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]
            
    return P_profile1, Q_profile1, P_profile2, Q_profile2


def ge_net2_profiles_load(net, INPUT_PATH, time_step, power_factor=0.95):
    """
    Function 5: To import load profiles for German net 2
    Demo: German net 2
    Versions:
        1st 27/02/2022
        2nd 23/04/2023 after repository update
    """
    # Import input profiles from German demo data
    # PATH_GE_NET2_PRF = IN_PATH+'/ge_profiles/ge_net2/ge_net2_profiles_base.xlsx'
    PATH_GE_NET2_PRF = os.path.join(INPUT_PATH, 'profile_files/ge_profiles/ge_net2/ge_net2_profiles_base.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_GE_NET2_PRF)
    net1_prf_sheet = net1_prf_book.sheets[0]
    net1_prf_input = net1_prf_sheet.range('A1').options(pd.DataFrame, index=True,header=True,expand='table').value.astype(float)
    net.load.profile_num = net.load.profile_num.astype(int)
    net1_prf_book.close()
    app.quit()
    
    # LOAD PROFILES BY LOAD INDEX
    # Empty profiles for P and Q
    P_profile = pd.DataFrame(np.zeros((time_step,len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    Q_profile = pd.DataFrame(np.zeros((time_step,len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    # Profiles assigment according to load profile type/number [Mwh]
    for i in tqdm(net.load.index):
        if (i==0 or i==1 or i==2 or i==150 or i==151): #MV loads Profiles
            max_coeff=(net1_prf_input.iloc[:,net.load.profile_num[i]]).max()
            max_p= net.load.p_mw[i]/max_coeff#to normalize
            max_q= net.load.q_mvar[i]/max_coeff#to normalize
            P_profile.iloc[:,i] = net1_prf_input.iloc[:,net.load.profile_num[i]]*max_p
            Q_profile.iloc[:,i] = net1_prf_input.iloc[:,net.load.profile_num[i]]*max_q
        elif (i>2 and (net.load.profile_type[i] == "Household" or net.load.profile_type[i] == "Commercial" or net.load.profile_type[i] == "Street_lighting")): #Household, Commercial, and Street lighting load Profiles
            P_profile.iloc[:,i] = (net1_prf_input.iloc[:,net.load.profile_num[i]]/1000)*net.load.profile_factor[i] #Only for ge_net2, multiplication of profile factor
            Q_profile.iloc[:,i] = np.sqrt((P_profile.iloc[:,i]/power_factor)**2 - P_profile.iloc[:,i]**2)
        elif(i>2 and (net.load.profile_type[i] == "Heatpump" or net.load.profile_type[i] == "Heatstorage")): #Heatstorage and Heatpump profiles, checl reactive power factor for heat storage!!!!!
            P_profile.iloc[:,i] = (net1_prf_input.iloc[:,net.load.profile_num[i]])*(net.load.kwh_heat[i]/1000) # multiply base normalized profile by MWh consumption
            #Q_profile.iloc[:,i] = np.sqrt((P_profile.iloc[:,i]/power_factor)**2 - P_profile.iloc[:,i]**2)
            Q_profile.iloc[:,i] = 0 #Because electrical heat storage
        else:
            P_profile.iloc[:,i] = "nan"
            Q_profile.iloc[:,i] = "nan"
    
    P_profile1 = P_profile.T
    Q_profile1 = Q_profile.T
    
    #LOAD PROFILES BY BUS INDEX
    # Empty profiles for P and Q
    P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    
    for i in tqdm(net.bus.index):
        for j in net.load.index:
            if net.load.bus[j] == i:
                P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
                Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]

    return P_profile1, Q_profile1, P_profile2, Q_profile2


def ge_net2_profiles_gen(net, INPUT_PATH, time_step):
    """
    Function 6: To import Generation Profiles for net2 German demo
    Demo: German
    Versions:
        1st 27/02/2022
        2nd 23/04/2023 after repository update
    """
    # Import input profiles from German demo data
    # PATH_GE_NET2_PRF = IN_PATH+'/ge_profiles/ge_net2/ge_net2_profiles_base.xlsx'
    PATH_GE_NET2_PRF= os.path.join(INPUT_PATH, 'profile_files/ge_profiles/ge_net2/ge_net2_profiles_base.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_GE_NET2_PRF)
    net1_prf_sheet = net1_prf_book.sheets[0]
    net1_prf_input = net1_prf_sheet.range('A1').options(pd.DataFrame, index=True,header=True,expand='table').value.astype(float)
    net.sgen.profile_num = net.sgen.profile_num.astype(int)
    net1_prf_book.close()
    app.quit()
    
    # GENERATION PROFILES BY GENERATOR INDEX
    # Empty profiles for P and Q considering the network values
    P_profile = pd.DataFrame(np.ones((time_step,len(net.sgen.index))), index=list(range(time_step)), columns=net.sgen.index)*np.nan
    Q_profile = pd.DataFrame(np.ones((time_step,len(net.sgen.index))), index=list(range(time_step)), columns=net.sgen.index)*np.nan
    
    # Multiply the empty profiles by the coefficients, according to the type of generator
    for i in tqdm(net.sgen.index):
        if net.sgen.profile_type[i] == 'PV':
            P_profile.iloc[:,i] = net.sgen.p_mw[i]*net1_prf_input.iloc[:,net.sgen.profile_num[i]]
            Q_profile.iloc[:,i] = net.sgen.q_mvar[i]*net1_prf_input.iloc[:,net.sgen.profile_num[i]]
        elif net.sgen.profile_type[i] == 'CHP':
            P_profile.iloc[:,i] = net.sgen.p_mw[i]*net1_prf_input.iloc[:,net.sgen.profile_num[i]]*0.75 #it is assumed a capacity factor 0f 75%
            Q_profile.iloc[:,i] = net.sgen.q_mvar[i]*net1_prf_input.iloc[:,net.sgen.profile_num[i]]*0.75
        else:
            P_profile.iloc[:,i] = "nan"
            Q_profile.iloc[:,i] = "nan"

            
    P_profile1 = P_profile.T
    Q_profile1 = Q_profile.T
    
    #GENERATION PROFILES BY BUS INDEX
    # Empty profiles for P and Q
    P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))
    Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))
    
    for i in tqdm(net.bus.index):
        for j in net.sgen.index:
            if net.sgen.bus[j] == i:
                P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
                Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]
            
    return P_profile1, Q_profile1, P_profile2, Q_profile2


def pt_net1_profiles_load_vMT(net, INPUT_PATH, time_step, power_factor=0.95):
    """
    Function 7: To import load profiles for Portugel net 1
    Demo: Portugal net 1
    Versions:
        1st 15/03/2023
        2nd 23/04/2023 after repository update
    """
    # Import input profiles from Portugal demo data
    # PATH_PT_NET1_PRF = os.path.join(INPUT_PATH,'profile_files/pt_profiles/pt_net1/pt_net1_profiles_base.xlsx')
    PATH_PT_NET1_PRF = os.path.join(INPUT_PATH, 'pt_net1_profiles_base.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_PT_NET1_PRF)
    net1_prf_sheet = net1_prf_book.sheets[0]  # Load profiles
    net1_prf_input = net1_prf_sheet.range('A1').options(pd.DataFrame, index=True, header=True, expand='table').value.astype(float)
    net1_prf_book.close()
    app.quit()
    # Import profile type from network file
    net.load.profile_type = net.load.profile_type.astype(int)
    
    # LOAD PROFILES BY LOAD INDEX
    # Empty profiles for P and Q
    P_profile = pd.DataFrame(np.zeros((time_step, len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    Q_profile = pd.DataFrame(np.zeros((time_step, len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    
    # Profiles assignment according to load profile_type from ERSE and annual consumption [MWh]
    for i in tqdm(net.load.index):
        if net.load.profile_type[i] == 1:
            # Applied c1 profile type, for MV loads
            P_profile.iloc[:, i] = net1_prf_input['c1']*net.load.annual_mwh[i]
            Q_profile.iloc[:, i] = np.sqrt((P_profile.iloc[:, i] / power_factor)**2 - P_profile.iloc[:, i]**2)
        elif net.load.profile_type[i] == 2:
            # Applied c2 profile type, for LV load with annual consumption more than 4140 MWh
            # (the load profile type was already assigned)
            P_profile.iloc[:, i] = net1_prf_input['c2']*net.load.annual_mwh[i]
            Q_profile.iloc[:, i] = np.sqrt((P_profile.iloc[:, i] / power_factor)**2 - P_profile.iloc[:, i]**2)
        elif net.load.profile_type[i] == 3:
            # Applied c3 profile type, for LV load with annual consumption less than 4140 MWh
            # (the load profile type was already assigned)
            P_profile.iloc[:, i] = net1_prf_input['c3']*net.load.annual_mwh[i]
            Q_profile.iloc[:, i] = np.sqrt((P_profile.iloc[:, i] / power_factor)**2 - P_profile.iloc[:, i]**2)
        else:
            P_profile.iloc[:, i] = "nan"
            Q_profile.iloc[:, i] = "nan"
    
    # P_profile1 = P_profile.T
    # Q_profile1 = Q_profile.T
    #
    # # LOAD PROFILES BY BUS INDEX
    # # Empty profiles for P and Q
    # P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index), time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    # Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index), time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    #
    # for i in tqdm(net.bus.index):
    #     for j in net.load.index:
    #         if net.load.bus[j] == i:
    #             P_profile2.iloc[i, :] = P_profile2.iloc[i, :] + P_profile1.iloc[j, :]
    #             Q_profile2.iloc[i, :] = Q_profile2.iloc[i, :] + Q_profile1.iloc[j, :]

    # for d in net.load.index:
    #     if d == 277:
    #         P_profile1.iloc[d, :] = P_profile1.iloc[d, :] + 1
    #         Q_profile1.iloc[d, :] = Q_profile1.iloc[d, :] + 0.32868
    #         net.load.sn_mva[d] = 1.05263  # Network MVA also modifi

    # return P_profile1, Q_profile1, P_profile2, Q_profile2
    return P_profile, Q_profile


def pt_net1_profiles_load(net, INPUT_PATH, time_step, power_factor=0.95):
    """
    Function 8: To import load profiles for Portugel net 1
    Demo: Portugal net 1
    Versions:
        1st 15/03/2023
        2nd 23/04/2023 after repository update
    """
    PATH_PT_NET1_PRF = os.path.join(INPUT_PATH, 'pt_net1_profiles_base.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_PT_NET1_PRF)
    net1_prf_sheet = net1_prf_book.sheets[0]  # Load profiles
    net1_prf_input = net1_prf_sheet.range('A1').options(pd.DataFrame, index=True, header=True, expand='table').value.astype(float)
    net1_prf_book.close()
    app.quit()
    # Import profile type from network file
    net.load.profile_type = net.load.profile_type.astype(int)
    
    # LOAD PROFILES BY LOAD INDEX
    # Empty profiles for P and Q
    P_profile = pd.DataFrame(np.zeros((time_step, len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    Q_profile = pd.DataFrame(np.zeros((time_step, len(net.load.index))), index=list(range(time_step)), columns=net.load.index)*np.nan
    
    # Profiles assignment according to load profile_type from ERSE and annual consumption [MWh]
    for i in tqdm(net.load.index):
        if net.load.profile_type[i] == 1:
            # Applied c1 profile type, for MV loads
            P_profile.iloc[:, i] = net1_prf_input['c1'] * net.load.annual_mwh[i]
            Q_profile.iloc[:, i] = np.sqrt((P_profile.iloc[:, i] / power_factor) ** 2 - P_profile.iloc[:, i] ** 2)
        elif net.load.profile_type[i] == 2:
            # Applied c2 profile type, for LV load with annual consumption more than 4140 MWh
            # (the load profile type was already assigned)
            P_profile.iloc[:, i] = net1_prf_input['c2'] * net.load.annual_mwh[i]
            Q_profile.iloc[:, i] = np.sqrt((P_profile.iloc[:, i] / power_factor) ** 2 - P_profile.iloc[:, i] ** 2)
        elif net.load.profile_type[i] == 3:
            # Applied c3 profile type, for LV load with annual consumption less than 4140 MWh
            # (the load profile type was already assigned)
            P_profile.iloc[:, i] = net1_prf_input['c3'] * net.load.annual_mwh[i]
            Q_profile.iloc[:, i] = np.sqrt((P_profile.iloc[:, i] / power_factor) ** 2 - P_profile.iloc[:, i] ** 2)
        else:
            P_profile.iloc[:, i] = "nan"
            Q_profile.iloc[:, i] = "nan"
    
    # P_profile1 = P_profile.T
    # Q_profile1 = Q_profile.T
    #
    # # LOAD PROFILES BY BUS INDEX
    # # Empty profiles for P and Q
    # P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index), time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    # Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index), time_step)), index=net.bus.index, columns=list(range(time_step)))*np.nan
    #
    # for i in tqdm(net.bus.index):
    #     for j in net.load.index:
    #         if net.load.bus[j] == i:
    #             P_profile2.iloc[i, :] = P_profile2.iloc[i, :] + P_profile1.iloc[j, :]
    #             Q_profile2.iloc[i, :] = Q_profile2.iloc[i, :] + Q_profile1.iloc[j, :]

    return P_profile, Q_profile


def pt_net1_profiles_gen(net, INPUT_PATH, time_step):
    """
    Function 9: To import Generation Profiles for net1 Portuguese demo
    Demo: Portugal
    Versions:
        1st 15/03/2023
        2nd 23/04/2023 after repository update
    """
    # Import input profiles from Portugal demo data
    PATH_PT_NET1_PRF = os.path.join(INPUT_PATH, 'pt_net1_profiles_base.xlsx')
    app = xw.App()
    net1_prf_book = xw.Book(PATH_PT_NET1_PRF)
    net1_prf_sheet = net1_prf_book.sheets[1]  # Generation profiles
    net1_prf_input = net1_prf_sheet.range('A1').options(pd.DataFrame, index=True,header=True,expand='table').value.astype(float)
    net1_prf_book.close()
    app.quit()

    #GENERATION PROFILES BY GENERATOR INDEX
    # Empty profiles for P and Q considering the network values
    P_profile = pd.DataFrame(np.ones((time_step,len(net.sgen.index))), index=list(range(time_step)), columns=net.sgen.index)*np.nan
    Q_profile = pd.DataFrame(np.ones((time_step,len(net.sgen.index))), index=list(range(time_step)), columns=net.sgen.index)*np.nan
    
    # Multiply the empty profiles by the coefficients, according to the type of generator
    for i in tqdm(net.sgen.index):
        # is the same profile because all pv generators have the same capacity and the base profiles consider that capacity (1.5kw)
        P_profile.iloc[:, i] = net1_prf_input["p_mw"]
        Q_profile.iloc[:, i] = P_profile.iloc[:, i]*0
            
    # P_profile1 = P_profile.T
    # Q_profile1 = Q_profile.T
    #
    # #GENERATION PROFILES BY BUS INDEX
    # # Empty profiles for P and Q
    # P_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))
    # Q_profile2 = pd.DataFrame(np.zeros((len(net.bus.index),time_step)), index=net.bus.index, columns=list(range(time_step)))
    #
    # for i in tqdm(net.bus.index):
    #     for j in net.sgen.index:
    #         if net.sgen.bus[j] == i:
    #             P_profile2.iloc[i,:] = P_profile2.iloc[i,:] + P_profile1.iloc[j,:]
    #             Q_profile2.iloc[i,:] = Q_profile2.iloc[i,:] + Q_profile1.iloc[j,:]
            
    return P_profile, Q_profile
