import os 

base_dir = f'/home/alex/drake-tamp/experiments/blocks_world/data_generation/perception_random/train'

def split_problems():
    i = 0 
    files_per_folder = 1000
    curr_path = ''
    for filename in os.listdir(base_dir): 
        f = os.path.join(base_dir, filename) 
        if os.path.isfile(f):
            if i % files_per_folder == 0: 
                curr_path = os.path.join(base_dir, '%03d'%(i/files_per_folder))
                try: 
                    os.makedirs(curr_path)
                except OSError:
                    print("dir already exists...")
            try: 
                os.replace(f, os.path.join(curr_path, filename))
            except OSError: 
                print("file already exists, skipping...")
            i = i + 1

def run_split_problems(): 
    os.chdir('/home/alex/drake-tamp/experiments/')
    tmux_session = 'collect-large'
    os.system('tmux new-session -d -s %s'%tmux_session)
    i = 0
    for filename in os.listdir(base_dir): 
        d = os.path.join(base_dir, filename)
        if os.path.isdir(d): 
            i = i + 1
            os.system('tmux new-window -t %s:%d'%(tmux_session, i))
            collect_command = './collect-labels.sh jobs/random-perception-large-01 blocks_world blocks_world/data_generation/perception_random/train/'+filename
            os.system('tmux send-keys -t %s:%d \'%s\' C-m'%(tmux_session, i,collect_command))
    return 

def test(): 
    os.chdir('/home/alex/drake-tamp/experiments/')
    os.system('tmux new-session -d -s test')
    os.system('tmux new-window -t test:4')
    os.system('tmux send-keys -t test:4 \'sleep 10\' C-m')
    os.system('tmux new-window -t test:2')
    os.system('tmux send-keys -t test:2 \'sleep 10\' C-m')


if __name__ == '__main__': 
    run_split_problems() 