import logging
import numpy as np
from predict_client.prod_client import ProdClient

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

# Make sure you have a model running on localhost:9000
host = '0.0.0.0:9000'
model_name = 'model'
model_version = 1

client = ProdClient(host, model_name, model_version)

req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': np.random.rand(10,2)}]

prediction = client.predict(req_data, request_timeout=10)
logger.info('Prediction: {}'.format(prediction))