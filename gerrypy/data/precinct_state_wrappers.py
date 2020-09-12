import os
from gerrypy import constants
from gerrypy.data.precincts import StatePrecinctWrapper

# TODO: add validation sources where available

class ALPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'AL'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'harvard_data', 'al_2010'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2008): ('USP_D_08', 'USP_R_08'),
                ('senate', 2008): ('USS_D_08', 'USS_R_08'),
            }
        }]

        self.county_inference = {
            ('pres', 2016): ('pres', 2008),
            ('pres', 2012): ('pres', 2008)
        }


class AKPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'AK'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'AK'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('pres', 2016): ('PRES16D', 'PRES16R')
            }
        }]

        self.county_inference = None

class AZPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'AZ'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'AZ'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('AG', 2018): ('AG18D', 'AG18R')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'az_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2016): ('G16USSDKIR', 'G16USSRMCC'),
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class ARPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'AR'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'ar_2016'),
            'precincts': True,
            'county_column': 'COUNTY_FIP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDELD', 'G16USSRBOO'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class CAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'CA'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'ca_2016'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('pres', 2016): ('G16PREDCli', 'G16PRERTru'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class COPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'CO'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'CO'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('AG', 2018): ('AG18D', 'AG18R')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'co_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('senate', 2016): ('G16USSDBen', 'G16USSRGle'),
                ('pres', 2016): ('G16PREDCli', 'G16PRERTru'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class CTPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'CT'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'CT'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('AG', 2018): ('AG18D', 'AG18R')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('gov', 2018),
            ('pres', 2012): ('gov', 2018)
        }


class DEPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'DE'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'DE'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('AG', 2018): ('AG18D', 'AG18R'),
                ('gov', 2016): ('GOV16D', 'GOV16R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2014): ('SEN14D', 'SEN14R'),
                ('AG', 2014): ('AG14D', 'AG14R'),
                ('gov', 2012): ('GOV12D', 'GOV12R'),
                ('pres', 2012): ('PRES12D', 'PRES12R'),
                ('senate', 2012): ('SEN12D', 'SEN12R'),
            }
        }]
        self.county_inference = {
            ('pres', 2008): ('pres', 2012),
        }


class FLPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'FL'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'fl_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCli', 'G16PRERTru'),
                ('senate', 2016): ('G16USSDMur', 'G16USSRRub'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class GAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'GA'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'GA'),
            'precincts': True,
            'county_column': 'FIPS2',
            'elections': {
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class HIPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'HI'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'HI'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R'),
            }
        }]
        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class IDPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'ID'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'id_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDSTU', 'G16USSRCRA'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class ILPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'IL'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'il_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDDUC', 'G16USSRKIR'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class INPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'IN'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'harvard_data', 'in_2010'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2008): ('OBAMA', 'MCCAIN'),
            }
        }]

        self.county_inference = {
            ('pres', 2016): ('pres', 2008),
            ('pres', 2012): ('pres', 2008)
        }


class IAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'IA'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'ia_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDJUD', 'G16USSRGRA'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class KSPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'KS'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'ks_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDWIE', 'G16USSRMOR'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class KYPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'KY'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'ky_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDGRA', 'G16USSRPAU'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class LAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'LA'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'la_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('R16USSDCAM', 'R16USSRKEN'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class MEPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'ME'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'me_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class MDPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MD'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'MD'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('AG', 2018): ('AG18D', 'AG18R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R'),
                ('gov', 2014): ('GOV14D', 'GOV14R'),
                ('AG', 2014): ('AG14D', 'AG14R'),
                ('pres', 2012): ('PRES12D', 'PRES12R'),
                ('senate', 2012): ('SEN12D', 'SEN12R'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2012)
        }


class MAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MA'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'MA12_16'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2014): ('SEN14D', 'SEN14R'),
                ('senate', 2013): ('SEN13D', 'SEN13R'),
                ('pres', 2012): ('PRES12D', 'PRES12R'),
                ('senate', 2012): ('SEN12D', 'SEN12R'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'MA02_10'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2010): ('SEN10D', 'SEN10R'),
                ('senate', 2008): ('SEN08D', 'SEN08R'),
                ('pres', 2008): ('PRES08D', 'PRES08R'),
            }
        }]
        self.county_inference = None


class MIPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MI'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'MI'),
            'precincts': True,
            'county_column': 'CountyFips',
            'elections': {
                ('pres', 2016): ('PRES16D', 'PRES16R'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class MNPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MN'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'MN'),
            'precincts': True,
            'county_column': 'COUNTYFIPS',
            'elections': {
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('AG', 2018): ('AG18D', 'AG18R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2014): ('SEN14D', 'SEN14R'),
                ('gov', 2014): ('GOV14D', 'GOV14R'),
                ('AG', 2014): ('AG14D', 'AG14R'),
                ('pres', 2012): ('PRES12D', 'PRES12R'),
                ('senate', 2012): ('SEN12D', 'SEN12R'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2012)
        }


class MSPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MS'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'harvard_data', 'ms_2010'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2008): ('USP_D_08', 'USP_R_08'),
                ('senate1', 2008): ('USS_1_D_08', 'USS_1_R_08'),
                ('senate2', 2008): ('USS_2_D_08', 'USS_R_2_08'),
            }
        }]

        self.county_inference = {
            ('pres', 2016): ('pres', 2008),
            ('pres', 2012): ('pres', 2008)
        }


class MOPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MO'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'mo_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDKAN', 'G16USSRBLU'),
                ('gov', 2016): ('G16GOVDKOS', 'G16GOVRGRE')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class MTPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MT'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'mt_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('gov', 2016): ('G16GOVDBUL', 'G16GOVRGIA'),
                ('AG', 2016): ('G16ATGDJEN', 'G16ATGRFOX'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class NEPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NE'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'ne_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class NVPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NV'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'nv_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDCOR', 'G16USSRHEC'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class NHPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NH'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'nh_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDHAS', 'G16USSRAYO'),
                ('gov', 2016): ('G16GOVDVAN', 'G16GOVRSUN')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class NJPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NJ'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'harvard_data', 'nj_2010'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2008): ('USP_DV_08', 'USP_RV_08'),
                ('senate', 2008): ('USS_DV_08', 'USS_RV_08'),
            }
        }]

        self.county_inference = {
            ('pres', 2016): ('pres', 2008),
            ('pres', 2012): ('pres', 2008)
        }


class NMPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NM'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'NM'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('AG', 2018): ('AG18D', 'AG18R'),
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
            }
        }]
        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class NYPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NY'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'harvard_data', 'ny_2010'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('gov', 2010): ('GOV_DVOTE_', 'GOV_RVOTE_'),
                ('AG', 2010): ('AG_DVOTE_1', 'AG_RVOTE_1'),
                ('senate1', 2010): ('USS_2_DVOT', 'USS_2_RVOT'),
                ('senate2', 2010): ('USS_6_DVOT', 'USS_6_RVOT')
            }
        }]

        self.county_inference = {
            ('pres', 2016): ('gov', 2010),
            ('pres', 2012): ('gov', 2010),
            ('pres', 2008): ('gov', 2010),
        }


class NCPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NC'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'NC'),
            'precincts': True,
            'county_column': 'County',
            'elections': {
                ('gov', 2016): ('EL16G_GV_D', 'EL16G_GV_R'),
                ('senate', 2016): ('EL16G_US_1', 'EL16G_USS_'),
                ('pres', 2016): ('EL16G_PR_D', 'EL16G_PR_R'),
                ('senate', 2014): ('EL14G_US_1', 'EL14G_USS_'),
                ('pres', 2012): ('EL12G_PR_D', 'EL12G_PR_R'),
                ('gov', 2012): ('EL12G_GV_D', 'EL12G_GV_R'),
                ('senate', 2010): ('EL10G_USS_', 'EL10G_US_1'),
                ('senate', 2008): ('EL08G_USS_', 'EL08G_US_1'),
                ('gov', 2008): ('EL08G_GV_D', 'EL08G_GV_R'),
            }
        }]
        self.county_inference = None


class NDPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'ND'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'nd_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDGLA', 'G16USSRHOE'),
                ('gov', 2016): ('G16GOVDNEL', 'G16GOVRBUR')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class OHPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'OH'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'OH'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class OKPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'OK'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'OK'),
            'precincts': True,
            'county_column': 'CNTYFIPS',
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('AG', 2018): ('AG18D', 'AG18R')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'ok_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2016): ('G16USSDWOR', 'G16USSRLAN'),
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class ORPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'OR'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'OR'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('gov', 2016): ('GOV16D', 'GOV16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('AG', 2016): ('AG16D', 'AG16R')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class PAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'PA'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'PA'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2016): ('T16PRESD', 'T16PRESR'),
                ('senate', 2016): ('T16SEND', 'T16SENR'),
                ('AG', 2016): ('T16ATGD', 'T16ATGR'),
                ('gov', 2014): ('F2014GOVD', 'F2014GOVR'),
                ('pres', 2012): ('PRES12D', 'PRES12R'),
                ('senate', 2012): ('USS12D', 'USS12R'),
                ('AG', 2012): ('ATG12D', 'ATG12R'),
                ('senate', 2010): ('SEN10D', 'SEN10R'),
                ('gov', 2010): ('GOV10D', 'GOV10R'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2012)
        }


class RIPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'RI'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'RI'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class SCPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'SC'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'sc_2016'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDDIX', 'G16USSRSCO'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class SDPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'SD'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'sd_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDWIL', 'G16USSRTHU'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class TNPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'TN'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'tn_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCli', 'G16PRERTru'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class TXPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'TX'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'TX'),
            'precincts': True,
            'county_column': 'FIPS',
            'elections': {
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('gov', 2014): ('GOV14D', 'GOV14R'),
                ('senate', 2014): ('SEN14D', 'SEN14R'),
                ('senate', 2012): ('SEN12D', 'SEN12R'),
                ('pres', 2012): ('PRES12D', 'PRES12R'),
            }
        }]
        self.county_inference = {
            ('pres', 2008): ('pres', 2012)
        }


class UTPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'UT'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'UT'),
            'precincts': True,
            'county_column': 'cnty_fp',
            'elections': {
                ('gov', 2016): ('GOV16D', 'GOV16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('gov', 2016),
            ('pres', 2012): ('gov', 2016)
        }


class VTPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'VT'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'VT'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R'),
                ('pres', 2012): ('PRES12D', 'PRES12R'),
                ('senate', 2012): ('SEN12B', 'SEN12R'),

            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2012),
        }


class VAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'VA'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'VA'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2018): ('G18DSEN', 'G18RSEN'),
                ('gov', 2017): ('G17DGOV', 'G17RGOV'),
                ('AG', 2017): ('G17DATG', 'G17RATG'),
                ('pres', 2016): ('G16DPRS', 'G16RPRS')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }

class WAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'WA'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'wa_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDMUR', 'G16USSRVAN'),
                ('gov', 2016): ('G16GOVDINS', 'G16GOVRBRY'),
                ('AG', 2016): ('G16ATGDFER', 'G16ATGRTRU')
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


class WVPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'WV'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError



class WIPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'WI'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'WI'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('PREDEM16', 'PREREP16'),
                ('senate', 2016): ('USSDEM16', 'USSREP16'),
                ('gov', 2014): ('GOVDEM14', 'GOVREP14'),
                ('AG', 2014): ('WAGDEM14', 'WAGREP14'),
                ('gov', 2012): ('GOVDEM12', 'GOVREP12'),
                ('pres', 2012): ('PREDEM12', 'PREREP12'),
                ('senate', 2012): ('USSDEM12', 'USSREP12'),
            }
        }]
        self.county_inference = {
            ('pres', 2008): ('pres', 2012)
        }

class WYPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'WY'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'wy_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
            }
        }]

        self.county_inference = {
            ('pres', 2008): ('pres', 2016),
            ('pres', 2012): ('pres', 2016)
        }


wrappers = {
    'AL': ALPrecinctWrapper,
    'AK': AKPrecinctWrapper,
    'AZ': AZPrecinctWrapper,
    'AR': ARPrecinctWrapper,
    'CA': CAPrecinctWrapper,
    'CO': COPrecinctWrapper,
    'CT': CTPrecinctWrapper,
    'DE': DEPrecinctWrapper,
    'FL': FLPrecinctWrapper,
    'GA': GAPrecinctWrapper,
    'HI': HIPrecinctWrapper,
    'ID': IDPrecinctWrapper,
    'IL': ILPrecinctWrapper,
    'IN': INPrecinctWrapper,
    'IA': IAPrecinctWrapper,
    'KS': KSPrecinctWrapper,
    'KY': KYPrecinctWrapper,
    'LA': LAPrecinctWrapper,
    'ME': MEPrecinctWrapper,
    'MD': MDPrecinctWrapper,
    'MA': MAPrecinctWrapper,
    'MI': MIPrecinctWrapper,
    'MN': MNPrecinctWrapper,
    'MS': MSPrecinctWrapper,
    'MO': MOPrecinctWrapper,
    'MT': MTPrecinctWrapper,
    'NE': NEPrecinctWrapper,
    'NV': NVPrecinctWrapper,
    'NH': NHPrecinctWrapper,
    'NJ': NJPrecinctWrapper,
    'NM': NMPrecinctWrapper,
    'NY': NYPrecinctWrapper,
    'NC': NCPrecinctWrapper,
    'ND': NDPrecinctWrapper,
    'OH': OHPrecinctWrapper,
    'OK': OKPrecinctWrapper,
    'OR': ORPrecinctWrapper,
    'PA': PAPrecinctWrapper,
    'RI': RIPrecinctWrapper,
    'SC': SCPrecinctWrapper,
    'SD': SDPrecinctWrapper,
    'TN': TNPrecinctWrapper,
    'TX': TXPrecinctWrapper,
    'UT': UTPrecinctWrapper,
    'VT': VTPrecinctWrapper,
    'VA': VAPrecinctWrapper,
    'WA': WAPrecinctWrapper,
    'WV': WVPrecinctWrapper,
    'WI': WIPrecinctWrapper,
    'WY': WYPrecinctWrapper,
}

if __name__ == '__main__':
    w = OKPrecinctWrapper()
    ps = w.load_precincts()
    shares = [w.compute_tract_results(p, i)[1] for i, p in enumerate(ps)]