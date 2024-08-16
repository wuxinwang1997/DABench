import os
import json
import moxing as mox

def openi_dataset_to_Env(data_url, data_dir):
    """
    openi copy single dataset to training image 
    """
    try:     
        mox.file.copy_parallel(data_url, data_dir)
        print("Successfully Download {} to {}".format(data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(data_url, data_dir) + str(e))
    return 

def openi_multidataset_to_env(multi_data_url, data_dir):
    """
    copy single or multi dataset to training image 
    """
    multi_data_json = json.loads(multi_data_url)  
    for i in range(len(multi_data_json)):
        path = data_dir + "/" + multi_data_json[i]["dataset_name"]
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            mox.file.copy_parallel(multi_data_json[i]["dataset_url"], path) 
            print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"],path))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], path) + str(e))
    return   

def pretrain_to_env(pretrain_url, pretrain_dir):
    """
    copy pretrain to training image
    """
    pretrain_url_json = json.loads(pretrain_url)  
    print("pretrain_url_json:",pretrain_url_json)
    for i in range(len(pretrain_url_json)):
        modelfile_path = pretrain_dir + "/" + pretrain_url_json[i]["model_name"]
        try:
            mox.file.copy(pretrain_url_json[i]["model_url"], modelfile_path) 
            print("Successfully Download {} to {}".format(pretrain_url_json[i]["model_url"],modelfile_path))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(pretrain_url_json[i]["model_url"], modelfile_path) + str(e))
    return          

def env_to_openi(train_dir, train_url):
    """
    upload output to openi 
    """
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank=int(os.getenv('RANK_ID'))
    if device_num == 1:
        obs_copy_folder(train_dir, train_url)
    if device_num > 1:
        if local_rank%8==0:
            obs_copy_folder(train_dir, train_url)
    return

def obs_copy_file(obs_file_url, file_url):
    """
    cope file from obs to obs, or cope file from obs to env, or cope file from env to obs
    """
    try:
        mox.file.copy(obs_file_url, file_url)
        print("Successfully Download {} to {}".format(obs_file_url,file_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_file_url, file_url) + str(e)) 
    return    
    
def obs_copy_folder(folder_dir, obs_folder_url):
    """
    copy folder from obs to obs, or copy folder from obs to env, or copy folder from env to obs
    """
    try:
        mox.file.copy_parallel(folder_dir, obs_folder_url)
        print("Successfully Upload {} to {}".format(folder_dir,obs_folder_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(folder_dir,obs_folder_url) + str(e))
    return     

def c2net_multidataset_to_env(multi_data_url, data_dir):
    """
    c2net copy single or multi dataset to training image 
    """
    multi_data_json = json.loads(multi_data_url)  
    for i in range(len(multi_data_json)):
        zipfile_path = data_dir + "/" + multi_data_json[i]["dataset_name"]
        try:
            mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path) 
            print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"],zipfile_path))
            #get filename and unzip the dataset
            filename = os.path.splitext(multi_data_json[i]["dataset_name"])[0]
            filePath = data_dir + "/" + filename
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            #If it is a tar compressed package, you can use os.system("tar -xvf {} {}".format(zipfile_path, filePath))
            os.system("unzip {} -d {}".format(zipfile_path, filePath))

        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], zipfile_path) + str(e))
    return       