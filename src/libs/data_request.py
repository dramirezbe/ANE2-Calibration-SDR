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
        
        # Unpack dictionaries directly into the new dataclasses
        return CampaignParams(
            name=data.get("name"),
            schedule=ScheduleParams(**data.get("schedule", {})),
            config=ConfigParams(**data.get("config", {}))
        )

    def get_api_signals(self, mac, camp_id):
        """Downloads all signal measurements for a MAC and campaign using pagination.
        
        Args:
            mac (str): MAC address of the sensor.
            camp_id (int): Campaign ID.
            
        Returns:
            list: List of raw measurement dictionaries.
        """
        #self._log.info(f"Starting signal download for MAC {mac}, Campaign {camp_id}")
        url = f"{self.base_url}/campaigns/sensor/{mac}/signals"
        params = {"campaign_id": camp_id, "page": 1, "page_size": 5000}
        signals = []
        while True:
            data = requests.get(url, params=params, verify=False).json()
            signals.extend(data['measurements'])
            if not data.get('pagination', {}).get('has_next'):
                break
            params['page'] += 1
        #self._log.info(f"Successfully downloaded {len(signals)} signals for MAC {mac}")
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
        df_full = {rbw: {} for rbw in campaigns}
        for rbw, camp_id in tqdm(campaigns.items(), desc="Campaigns"):
            for node_id in tqdm(node_ids, desc="Nodes", leave=False):
                mac = self.node_macs.get(node_id)
                if mac:
                    datos = self.get_api_signals(mac, camp_id)
                    if datos:
                        df_full[rbw][f"Node{node_id}"] = pd.DataFrame(datos).dropna(subset=['pxx'])
        self._log.info("Finished loading data for all campaigns and nodes.")
        return df_full