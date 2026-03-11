import cfg
log = cfg.set_logger()
from libs.data_request import DataRequest

def main():
    log.info(f"Starting {cfg.APP_NAME} v{cfg.APP_VERSION} in {cfg.COUNTRY}...")
    dr = DataRequest(log=log, base_url=cfg.API_URL)

    log.info(f"Object class DataRequest print = {dr}")

if __name__ == "__main__":
    main()