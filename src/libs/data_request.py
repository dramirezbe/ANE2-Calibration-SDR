import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

@dataclass
class ScheduleParams:
    start_date: str
    end_date: str
    start_time: str
    end_time: str
    interval_seconds: int

@dataclass
class ConfigParams:
    rbw: str
    span: int
    antenna: str
    lna_gain: int
    vga_gain: int
    antenna_amp: bool
    center_freq_hz: int
    sample_rate_hz: int
    centerFrequency: int

@dataclass
class CampaignParams:
    name: str
    schedule: ScheduleParams
    config: ConfigParams
    pxx_len: int

class DataRequest:
    """Handles data requests from the API.
    
    Args:
        log (logging.Logger, optional): Logger instance. Defaults to None.
        base_url (str, optional): Base API URL. Defaults to ANE endpoint.
    """
    def __init__(self, log=None,base_url="https://rsm.ane.gov.co:12443/api"):
        self.base_url = base_url

        if not log:
            import logging
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
            self._log = logging.getLogger("DataRequest")
        else:
            self._log = log

        self._log.info(f"Initialized DataRequest class with base_url: {self.base_url}")
        
        self.node_macs = {
            1: 'd8:3a:dd:f7:1d:f2', 2: 'd8:3a:dd:f4:4e:26', 3: 'd8:3a:dd:f7:22:87',
            4: 'd8:3a:dd:f6:fc:be', 5: 'd8:3a:dd:f7:21:52', 6: 'd8:3a:dd:f7:1a:cc',
            7: 'd8:3a:dd:f7:1d:b6', 8: 'd8:3a:dd:f7:1b:20', 9: 'd8:3a:dd:f4:4e:d1',
            10: 'd8:3a:dd:f7:1d:90'
        }

    def get_realtime_signal(self, node_id):
        """Fetches real-time signal data for a specific node.
        
        Args:
            node_id (int): The ID of the node (1-10).
            
        Returns:
            RealtimeSignal: Object containing the parsed signal data.
        """
        mac = self.node_macs.get(node_id)
        self._log.info(f"Fetching realtime signal for Node {node_id} (MAC: {mac})")
        url = f"{self.base_url}/campaigns/sensor/{mac}/realtime"
        data = requests.get(url, verify=False).json()
        
        sig_entry = data['signal'][0] if data.get('signal') else {}
        
        return RealtimeSignal(
            mac=data.get("mac"),
            active_configuration=data.get("active_configuration"),
            pxx=np.array(sig_entry.get("pxx", [])),
            start_freq_hz=sig_entry.get("start_freq_hz"),
            end_freq_hz=sig_entry.get("end_freq_hz")
        )

    def get_campaign_params(self, campaign_id):
        """Retrieves parameters for a specific campaign.
        
        Args:
            campaign_id (int): The ID of the campaign.
            
        Returns:
            CampaignParams: Object containing the campaign's parameters.
        """
        self._log.info(f"Fetching parameters for campaign ID: {campaign_id}")
        url = f"{self.base_url}/campaigns/{campaign_id}/parameters"
        data = requests.get(url, verify=False).json()
        
        config_data = data.get("config", {})
        
        # --- Calculate pxx_len ---
        # 1. Safely parse RBW (it comes in as a string according to your logs)
        try:
            rbw_val = float(config_data.get("rbw", 1000.0))
            safe_rbw = rbw_val if rbw_val > 0 else 1000.0
        except (ValueError, TypeError):
            safe_rbw = 1000.0
            
        # 2. Get sample rate
        sample_rate = float(config_data.get("sample_rate_hz", 0.0))
        
        # 3. Apply the C-formula logic
        if sample_rate > 0:
            enbw_factor = 1.363 # HAMMING_TYPE
            required_nperseg_val = enbw_factor * sample_rate / safe_rbw
            exponent = int(np.ceil(np.log2(required_nperseg_val)))
            pxx_len = int(2 ** exponent)
        else:
            pxx_len = 0 # Fallback if no sample rate is provided
        # -------------------------

        # Unpack dictionaries directly into the new dataclasses
        return CampaignParams(
            name=data.get("name"),
            schedule=ScheduleParams(**data.get("schedule", {})),
            config=ConfigParams(**config_data),
            pxx_len=pxx_len
        )

    def get_api_signals(self, mac, camp_id, node_desc):
        """
        Fetches signal measurements for a given sensor MAC address and campaign ID, handling pagination.
        Args:
            mac (str): The MAC address of the sensor.
            camp_id (int): The campaign ID to filter signals.
            node_desc (str): Description of the node for logging purposes.
        Returns:
            list: A list of signal measurements retrieved from the API.
        """
        url = f"{self.base_url}/campaigns/sensor/{mac}/signals"
        params = {"campaign_id": camp_id, "page": 1, "page_size": 1000}
        signals = []
        
        # Single, flat progress bar that works everywhere
        with tqdm(desc=f"  ↳ {node_desc}", unit="page", leave=True) as pbar:
            while True:
                try:
                    response = requests.get(url, params=params, verify=False, timeout=20)
                    data = response.json()
                    
                    measurements = data.get('measurements', [])
                    if not measurements:
                        break
                        
                    signals.extend(measurements)
                    pbar.update(1)
                    
                    if not data.get('pagination', {}).get('has_next'):
                        break
                    
                    params['page'] += 1
                    
                except Exception as e:
                    print(f"\n  [!] Timeout/Error on {node_desc} (MAC {mac}): {e}")
                    break 
            
        return signals

    def load_campaigns_and_nodes(self, campaigns, node_ids):
        """Loads signals for multiple campaigns and nodes into pandas DataFrames.
        
        Args:
            campaigns (dict): Dictionary mapping string labels to campaign IDs.
            node_ids (list): List of node IDs to process.
            
        Returns:
            dict: Nested dictionary containing DataFrames of signals per campaign/node.
        """
        self._log.info(f"Loading data for campaigns: {list(campaigns.keys())} and nodes: {node_ids}")
        df_full = {camp: {} for camp in campaigns}
        
        # Removed tqdm here to prevent nesting bugs in Jupyter
        for camp, camp_id in campaigns.items():
            print(f"\n🚀 Starting Campaign: {camp}")
            for node_id in node_ids:
                mac = self.node_macs.get(node_id)
                if mac:
                    datos = self.get_api_signals(mac, camp_id, node_desc=f"Node {node_id}")
                    if datos:
                        df_full[camp][f"Node{node_id}"] = pd.DataFrame(datos).dropna(subset=['pxx'])
                        
        self._log.info("Finished loading data for all campaigns and nodes.")
        return df_full