

import configparser
import argparse
import torch
import numpy as np

from models import MLP,create_tf_model

from Problems import init_problem

import tensorflow as tf
from datetime import datetime
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change config filename based on input.")
    
    # Specify a default value for the 'input' argument.
    parser.add_argument("-c", "--config", 
                        help="Input to change the config name.", 
                        default="../configs/pygranso.cfg")
    parser.add_argument("-rho", "--rho", type=float,
                    help="RHO value for penalty method.", 
                    default=None)
    parser.add_argument("-cce", "--cce", type=float,
                help="CHECK_CONSTR_EVERY.", 
                default=None)
    parser.add_argument("-rf", "--rho_factor", type=float,
                    help="RHO factor", 
                    default=None)
    parser.add_argument("-t", "--threshold", type=float,
                help="Vio tolerance threshold for the constraint", 
                default=None)

    args = parser.parse_args()
    print("Loading cfgs")

    cfg = configparser.ConfigParser()
    #cp.read('./config/config_mnist.cfg')
    cfg.read(args.config)
    if args.rho is not None:
        cfg.set('OPTIMIZER','RHO',str(args.rho))
    if args.threshold is not None:
        cfg.set('EXP','THRESHOLD',str(args.threshold))
    if args.rho_factor is not None:
        cfg.set('OPTIMIZER','RHO_FACTOR',str(args.rho_factor))
    if args.cce is not None:
        cfg.set('OPTIMIZER','CHECK_CONSTR_EVERY',str(args.cce))
     

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    BASE_FOLDER = cfg.get('EXP','FOLDER')
    

    PROB_NAME = cfg.get('OPTIMIZER','NAME')
    DATASET_NAME = cfg.get('DATASET','NAME')
    DATASET_PATH = cfg.get('DATASET','PATH')
    npz_data = np.load(DATASET_PATH)

    now = datetime.now()

    # Format the date and time as a string
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    print(f'current time is {date_time_str}')
    NUM_FEATURES = npz_data['train_X_0'].shape[1]
    THRESHOLD = cfg.get('EXP','THRESHOLD')
    BN = cfg.getboolean('MODEL','BN')
    if PROB_NAME == 'PENALTY':
        penalty_type = cfg.get('OPTIMIZER','TYPE')
        RHO = cfg.get('OPTIMIZER','RHO')
        CCE = cfg.get('OPTIMIZER','CHECK_CONSTR_EVERY')
        RHO_FACTOR = cfg.get('OPTIMIZER','RHO_FACTOR')
        EXP_FOLDER = os.path.join(BASE_FOLDER,'exp',f'r_{THRESHOLD}',
                                  f'{PROB_NAME}_{DATASET_NAME}_RHO_{RHO}'+
                                  f'_CHECK_CONSTR_EVERY_{CCE}_RHO_FACTOR_{RHO_FACTOR}_PT_{penalty_type}_BN_{BN}')
    elif PROB_NAME == 'PyGRANSO_PENALTY':
        RHO = cfg.get('OPTIMIZER','RHO')
        EXP_FOLDER = os.path.join(BASE_FOLDER,'exp',f'r_{THRESHOLD}',
                            f'{PROB_NAME}_{DATASET_NAME}_RHO_{RHO}_BN_{BN}')
    else:    
        EXP_FOLDER = os.path.join(BASE_FOLDER,'exp',f'r_{THRESHOLD}',f'{PROB_NAME}_{DATASET_NAME}_BN_{BN}')
    PLATFORM = cfg.get('MODEL','PLATFORM')
    os.makedirs(EXP_FOLDER,exist_ok=True)

    cfg_path = os.path.join(EXP_FOLDER,'config.cfg')
    print(f'Saving current cfg to {cfg_path}')
    with open(cfg_path, 'w') as configfile:
        cfg.write(configfile)

    if PLATFORM == 'PyTorch':

        END_REPEAT = cfg.getint('EXP','END_REPEAT')
        START_REPEAT = cfg.getint('EXP','START_REPEAT')

        MODEL_TYPE = cfg.get('MODEL','TYPE')
        LAYER_WIDTH = cfg.getint('MODEL','LAYER_WIDTH')
        NUM_LAYERS = cfg.getint('MODEL','NUM_LAYERS')
        MODEL_PATH = cfg.get('MODEL','PATH')
        MODEL_OUT_ACTIVATION = cfg.get('MODEL','OUT_ACTIVATION')
        
        
        data = {key: torch.tensor(npz_data[key]) for key in npz_data.files}

        print('\n'+'='*30)
        print('Building pytorch model')
        DIR_NAME = f'{MODEL_TYPE}_{NUM_LAYERS}_Feature{NUM_FEATURES}'
        if not BN:
            DIR_NAME += '_No_BN'
        MODEL_DIR = os.path.join(MODEL_PATH,DIR_NAME)

        model = MLP(1,NUM_FEATURES,NUM_LAYERS,LAYER_WIDTH,BN,MODEL_OUT_ACTIVATION)
        for i in range(START_REPEAT,END_REPEAT):
            print('\n'+'='*30)
            print(f'EXP {i} Start!!!')
            weight_path = os.path.join(MODEL_DIR,f'model_pytorch_{i}.pt')
            fn = os.path.join(EXP_FOLDER,f'{i:06}.npz')
            print(f'>Results will be save to {fn}')
            print(f'>Loading weight from: {weight_path}')
            model.load_state_dict(torch.load(weight_path))
            prob = init_problem(cfg,data,device,model,fn=fn)
            prob.train()

    elif PLATFORM == 'Tensorflow':

        END_REPEAT = cfg.getint('EXP','END_REPEAT')
        START_REPEAT = cfg.getint('EXP','START_REPEAT')

        MODEL_TYPE = cfg.get('MODEL','TYPE')
        LAYER_WIDTH = cfg.getint('MODEL','LAYER_WIDTH')
        NUM_LAYERS = cfg.getint('MODEL','NUM_LAYERS')
        MODEL_PATH = cfg.get('MODEL','PATH')
        MODEL_OUT_ACTIVATION = cfg.get('MODEL','OUT_ACTIVATION')
        
        #data = {key: torch.tensor(npz_data[key]) for key in npz_data.files}

        print('\n'+'='*30)
        #print('Building Model')
        print('Building tensorflow model')
        DIR_NAME = f'{MODEL_TYPE}_{NUM_LAYERS}_Feature{NUM_FEATURES}'
        if not BN:
            DIR_NAME += '_No_BN'
        MODEL_DIR = os.path.join(MODEL_PATH,DIR_NAME)

        #model = create_tf_model(1,NUM_FEATURES,NUM_LAYERS,LAYER_WIDTH,MODEL_OUT_ACTIVATION)
        
        for i in range(START_REPEAT,END_REPEAT):
            print('\n'+'='*30)
            print(f'EXP {i} Start!!!')
            weight_path = os.path.join(MODEL_DIR,f'model_keras_{i}.h5')
            fn = os.path.join(EXP_FOLDER,f'{i:06}.npz')
            print(f'>Results will be save to {fn}')
            print(f'>Loading weight from: {weight_path}')
            #model = tf.saved_model.load(weight_path)
            model = tf.keras.models.load_model(weight_path)
            #model.summary()
            #model.load_weights(weight_path)
            prob = init_problem(cfg,npz_data,device,model,fn=fn)
            prob.train()




    

