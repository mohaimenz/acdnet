import os
import requests
from common.constants import data_support

def get_data_path(set_name, set_type):
    if set_name not in data_support:
        raise Exception(f'Download failed. Set not known, {set_name}')
    
    if set_type not in data_support[set_name]:
        return None

    # cwd_path = os.path.dirname(os.path.abspath(__file__))
    base_path = data_support[set_name]['path']
    file_path = data_support[set_name][set_type]['file_name']
    # return os.path.join(cwd_path, base_path, file_path)
    return os.path.join(base_path, file_path)

def download(set_name):
    def download_file(set_name, set_type):
        if set_name not in data_support:
            raise Exception(f'Download failed. Set not known, {set_name}')
    
        if set_type not in data_support[set_name]:
            return None

        # get url
        data_set = data_support[set_name][set_type]
        url = data_set['url']

        # get and prepare destination
        data_path = get_data_path(set_name, set_type)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        if os.path.exists(data_path):
            print(f'Skipping download {url}. Path exists {data_path}')
            return

        print(f'Downloading {data_set["url"]} -> {data_path}')

        # Get the Google drive cookie, accept it by returning 
        # the confirm value on subsequent request
        session = requests.Session()
        response = session.get(url, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(url, params = params, stream = True)

        # Save file to disk
        save_response_content(response, data_path)

    download_file(set_name, 'test')
    download_file(set_name, 'representative')