import os

def get_params():

    params = {}
    
    # Parameters
    params['dataset'] = 'QIK'
    
    params['stage']= 'rerank' # if 'rerank', display will include detections. if 'rerank2nd', reranking will be performed with QE top N locations
    params['use_regressed_boxes'] = False
    params['use_class_scores'] = False
    params['gpu'] = True # Applies to feature extraction and reranking
    params['distance'] = 'cosine'
    params['pooling'] = 'max'
    
    params['fast_rcnn_path'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/py-faster-rcnn/'
    params['net'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/models/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'
    params['net_proto'] ='/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/models/test.prototxt'
            
    params['layer'] = 'conv5_3'
    params['dimension'] = 512 # number of filters in the used layer
      
    params['K'] = 100000 # Number of elements to consider for mAP (high number means all elements are evaluated)
    params['num_rerank'] = 100 # Top elements to consider for reranking
    params['size_box'] = 15 # for display 
    params['N_QE'] = 5 # number of elements for query expansion
    params['N_display'] = 10 # top N ranking to display
    params['figsize'] = (80,40)
    
    
    if params['dataset'] is 'oxford':
        
        params['query_names'] = ["all_souls", "ashmolean", "balliol","bodleian", "christ_church", "cornmarket","hertford","keble","magdalen","pitt_rivers","radcliffe_camera"]
        
        params['database_images'] = 'data/images/oxford/data/' # oxford
        params['ground_truth_file'] = 'data/images/oxford/groundtruth'
        
    elif params['dataset'] is 'paris':
        
        params['query_names'] = ["defense", "eiffel","invalides", "louvre", "moulinrouge","museedorsay","notredame","pantheon","pompidou","sacrecoeur", "triomphe"]
        
        params['database_images'] = 'data/images/paris/data/paris' # paris
        params['ground_truth_file'] = 'data/images/paris/groundtruth'

    elif params['dataset'] is 'QIK':
        params['database_images'] = '/mydata/apache-tomcat/webapps/QIK_Image_Data/'
    
    params['query_list'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/imagelists/query' +'_' +params['dataset'] +  '.txt' # A txt file  
    params['frame_list'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/imagelists/' + params['dataset'] + '.txt' # A txt file - trecvid

    # Storage
    params['rankings_dir'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/rankings/'+ params['dataset']
    

    params['database_feats'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/features/' + params['dataset'] + '_' + params['layer'] + '_' + params['pooling']+ '.pkl' # This is a single pickle file
    params['pca_model'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/pca/' + params['layer'] + '_' + params['pooling'] 
 
    params['figures_path'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/figures/'
    params['reranking_path'] = '/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/reranking/'
    
    params['paris_corrupt_list'] = ['paris_louvre_000136.jpg',
                                    'paris_louvre_000146.jpg',
                                    'paris_moulinrouge_000422.jpg', 
                                    'paris_museedorsay_001059.jpg',
                                    'paris_notredame_000188.jpg',
                                    'paris_pantheon_000284.jpg',
                                    'paris_pantheon_000960.jpg',
                                    'paris_pantheon_000974.jpg',
                                    'paris_pompidou_000195.jpg',
                                    'paris_pompidou_000196.jpg',
                                    'paris_pompidou_000201.jpg',
                                    'paris_pompidou_000467.jpg',
                                    'paris_pompidou_000640.jpg',
                                    'paris_sacrecoeur_000299.jpg',
                                    'paris_sacrecoeur_000330.jpg',
                                    'paris_sacrecoeur_000353.jpg',
                                    'paris_triomphe_000662.jpg',
                                    'paris_triomphe_000833.jpg',
                                    'paris_triomphe_000863.jpg',
                                    'paris_triomphe_000867.jpg']
    
    
    list_of_dirs = ['/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/imagelists','/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/rankings/','/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/features','/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/rankings','/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/rankings/oxford','/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/rankings/paris','/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/reranking','/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/pca','/mydata/QIK-IR-ClipCap/ML_Models/DeepVision/data/figures']
    
    for _dir in list_of_dirs:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)    
            
    return params


if __name__ == "__main__":
    params = get_params()