import os
from gerrypy import constants
from gerrypy.data.precincts import StatePrecinctWrapper

# TODO: add validation sources where available

class ALPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'AL'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class AKPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'AK'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'AK'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('senate', 2016): ('SEN16D', 'SEN16R'),
                ('pres', 2016): ('PRES16D', 'PRES16R')
            }
        }]


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
            ('pres', 2016): ('pres', 2008),
            ('pres', 2016): ('pres', 2012)
        }


class ARPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'AR'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class CAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'CA'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class COPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'CO'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'CO'),
            'precincts': True,
            'county_column': 'COUNTY',
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
            ('pres', 2016): ('pres', 2008),
            ('pres', 2016): ('pres', 2012)
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
            ('gov', 2018): ('pres', 2008),
            ('gov', 2018): ('pres', 2012)
        }


class DEPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'DE'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class FLPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'FL'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


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
            ('pres', 2016): ('pres', 2012),
            ('pres', 2016): ('pres', 2008)
        }


class HIPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'HI'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class IDPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'ID'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class ILPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'IL'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class INPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'IN'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class IAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'IA'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class KSPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'KS'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class KYPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'KY'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class LAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'LA'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class MEPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'ME'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


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
            ('pres', 2012): ('pres', 2008)
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
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('senate', 2018): ('SEN18D', 'SEN18R'),
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R'),
                ('gov', 2014): ('GOV14D', 'GOV14R'),
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
            ('pres', 2016): ('pres', 2012),
            ('pres', 2016): ('pres', 2008),
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
            ('pres', 2012): ('pres', 2008)
        }


class MSPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MS'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class MOPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MO'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class MTPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'MT'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class NEPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NE'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class NVPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NV'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class NHPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NH'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class NJPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NJ'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


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
            ('pres', 2016): ('pres', 2008),
            ('pres', 2016): ('pres', 2012)
        }


class NYPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NY'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class NCPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'NC'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'NC'),
            'precincts': True,
            'county_column': 'County',
            'elections': {
                ('gov', 2016): ('EL16_GV_D', 'EL16_GV_R'),
                ('senate', 2016): ('EL16_US_1', 'EL16_USS_'),
                ('pres', 2016): ('EL16_PR_D', 'EL16_PR_R'),
                ('senate', 2014): ('EL14_USS_', 'EL14_US_1'),
                ('pres', 2012): ('EL12_PR_D', 'EL12_PR_R'),
                ('gov', 2012): ('EL12_GV_D', 'EL12_GV_R'),
                ('senate', 2010): ('EL10_USS_', 'EL10_US_1'),
                ('senate', 2008): ('EL08_USS_', 'EL08_US_1'),
                ('gov', 2008): ('EL08_GV_D', 'EL08_GV_R'),
            }
        }]


class NDPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'ND'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class OHPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'OH'
        self.main_source = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'OH'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('pres', 2016): ('PRES16D', 'PRES16R'),
                ('senate', 2016): ('SEN16D', 'SEN16R')
            }
        }]

        self.county_inference = {
            ('pres', 2012): ('pres', 2008)
        }


class OKPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'OK'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'mggg_states', 'OK'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('gov', 2018): ('GOV18D', 'GOV18R'),
                ('AG', 2018): ('AG18D', 'AG18R')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'fsu_data', 'ok_2016'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('senate', 2016): ('G16USSDBen', 'G16USSRGle'),
                ('pres', 2016): ('G16PREDCli', 'G16PRERTru'),
            }
        }]

        self.county_inference = {
            ('pres', 2016): ('pres', 2008),
            ('pres', 2016): ('pres', 2012)
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
            ('pres', 2016): ('pres', 2008),
            ('pres', 2016): ('pres', 2012)
        }


class PAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'PA'
        self.main_source = {
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
        }

        self.county_inference = {
            ('pres', 2012): ('pres', 2008)
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
            ('pres', 2016): ('pres', 2012),
            ('pres', 2016): ('pres', 2008),
        }


class SCPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'SC'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class SDPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'SD'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


class TNPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'TN'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


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
            ('gov', 2012): ('pres', 2012)
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
            ('gov', 2016): ('pres', 2008),
            ('gov', 2016): ('pres', 2012)
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
                ('senate', 2012): ('SEN16B', 'SEN16R'),

            }
        }]

        self.county_inference = {
            ('pres', 2012): ('pres', 2008),
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
                ('senate', 2016): ('SEN16D', 'SEN16R'),
                ('pres', 2016): ('G16DPRS', 'G16RPRS')
            }
        }]

        self.county_inference = {
            ('pres', 2016): ('pres', 2008),
            ('pres', 2016): ('pres', 2012)
        }


class WAPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'WA'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError


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
                ('AG', 2012): ('WAGDEM12', 'WAGREP12'),
            }
        }]
        self.county_inference = {
            ('pres', 2012): ('pres', 2008)
        }

class WYPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'WY'
        self.main_sources = [{}]
        self.county_inference = {}
        raise NotImplementedError