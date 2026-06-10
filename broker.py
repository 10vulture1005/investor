import os
import logging
from dotenv import load_dotenv

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

logger = logging.getLogger(__name__)

class DhanBroker:
    def __init__(self):
        load_dotenv()
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        self.dhan = None
        
        if self.client_id and self.access_token and dhanhq:
            try:
                self.dhan = dhanhq(self.client_id, self.access_token)
                logger.info("DhanHQ API Client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize DhanHQ Client: {e}")
        else:
            logger.warning("Dhan API keys not found in .env. Running in offline/mock mode.")
            
    def is_connected(self):
        return self.dhan is not None
        
    def get_live_capital(self):
        """
        Fetches the live available trading margin from Dhan.
        Returns the float value if successful, or None if offline/failed.
        """
        if not self.is_connected():
            return None
            
        try:
            fund_data = self.dhan.get_fund_limits()
            if fund_data.get('status') == 'success':
                # Parse available margin. The exact key might vary depending on API version.
                # Usually it's in data -> availabelBalance or similar.
                data = fund_data.get('data', {})
                avail_margin = data.get('availabelBalance', data.get('availableBalance', 0.0))
                return float(avail_margin)
            else:
                logger.error(f"Failed to fetch fund limits: {fund_data}")
                return None
        except Exception as e:
            logger.error(f"Exception fetching live capital: {e}")
            return None
