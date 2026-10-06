from pathlib import Path

LATENT_DIM = 64
IN_CHANNELS = 1
EPOCHS = 20              
LEARNING_RATE = 1e-4
BETA = 1e-6  #1e-4
KL_REDUCTION = "sum"    
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0      
BATCH_SIZE = 16           
EVAL_BATCH_SIZE = 16     
SEED = 42                
SPLIT_SEED = 42           

UP2_PATH = Path("data/718RX_1um_120x120.up2")
UP2_OFFSET = 16           
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_FILE = None        
MAX_PATTERNS = 60000      
NORMALIZATION = "per_pattern_minmax" 

DEVICE = "auto"          
NUM_WORKERS = 0          
PREFETCH_FACTOR = 1       
PERSISTENT_WORKERS = True  
PIN_MEMORY = "auto"       
CPU_THREADS = None        
LOG_EVERY_STEPS = 100     
CHECK_FINITE_EVERY = 1     
PROGRESS_MININTERVAL = 1.0 

OUTPUT_ROOT = Path("runs")
RUN_NAME = "vae"          
PLOT_EVERY_EPOCHS = 1
RECON_EVERY_EPOCHS = 1    
NUM_RECON_IMAGES = 8
PLOT_DPI = 140
RUN_TEST_AT_END = True    
RESUME = None            
