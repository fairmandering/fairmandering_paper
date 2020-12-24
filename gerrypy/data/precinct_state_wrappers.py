import os
from gerrypy import constants
from gerrypy.data.precincts import StatePrecinctWrapper

# TODO: add validation in elections with duplicate sources


class ALPrecinctWrapper(StatePrecinctWrapper):
    def __init__(self):
        super().__init__()
        self.state = 'AL'
        self.main_sources = [{
            'path': os.path.join(constants.PRECINCT_PATH, 'ansolabehere_rodden_2010', 'al_2010'),
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
            }, 
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'az_2016'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'ar_2016'),
            'precincts': True,
            'county_column': 'COUNTY_FIP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDELD', 'G16USSRBOO'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'ar_2018'),
            'precincts': True,
            'county_column': 'COUNTY_FIP',
            'elections': {
                ('gov', 2018): ('G18GOVDHEN', 'G18GOVRHUT'),
                ('AG', 2018): ('G18ATGDLEE', 'G18ATGRRUT')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'ca_2016'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'co_2016'),
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
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'de_2018'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('senate', 2018): ('G18USSDCAR', 'G18USSRARL'),
                ('AG', 2018): ('G18ATGDJEN', 'G18ATGRPEP')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'fl_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCli', 'G16PRERTru'),
                ('senate', 2016): ('G16USSDMur', 'G16USSRRub'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'fl_2018'),
            'precincts': True,
            'county_column': 'County',
            'elections': {
                ('gov', 2018): ('G18GOVDGIL', 'G18GOVRDES'),
                ('AG', 2018): ('G18ATGDSHA', 'G18ATGRMOO')
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
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'ga_2018'),
            'precincts': True,
            'county_column': 'FIPS2',
            'elections': {
                ('gov', 2018): ('G18GOVDABR', 'G18GOVRKEM'),
                ('AG', 2018): ('G18ATGDBAI', 'G18ATGRCAR')
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
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'hi_2018'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('senate', 2018): ('G18USSDHIR', 'G18USSRCUR'),
                ('gov', 2018): ('G18GOVDIGE', 'G18GOVRTUP')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'id_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDSTU', 'G16USSRCRA'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'id_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('AG', 2018): ('G18ATGDBIS', 'G18ATGRWAS'),
                ('gov', 2018): ('G18GOVDJOR', 'G18GOVRLIT')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'il_2016'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'ansolabehere_rodden_2010', 'in_2010'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'ia_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDJUD', 'G16USSRGRA'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'ia_2018'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('gov', 2018): ('G18GOVDHUB', 'G18GOVRREY')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'ks_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDWIE', 'G16USSRMOR'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'ks_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('AG', 2018): ('G18ATGDSWA', 'G18ATGRSCH'),
                ('gov', 2018): ('G18GOVDKEL', 'G18GOVRKOB')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'ky_2016'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'la_2016'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'me_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'me_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('gov', 2018): ('G18GOVDMIL', 'G18GOVRMOO')
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
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'ma_2018'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('AG', 2018): ('G18ATGDHEA', 'G18ATGRMCM'),
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
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'mi_2018'),
            'precincts': True,
            'county_column': 'COUNTYFIPS',
            'elections': {
                ('senate', 2018): ('G18USSDSTA', 'G18USSRJAM'),
                ('AG', 2018): ('G18ATGDNES', 'G18ATGRLEO'),
                ('gov', 2018): ('G18GOVDWHI', 'G18GOVRSCH')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'ansolabehere_rodden_2010', 'ms_2010'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'mo_2016'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'mt_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('gov', 2016): ('G16GOVDBUL', 'G16GOVRGIA'),
                ('AG', 2016): ('G16ATGDJEN', 'G16ATGRFOX'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'mt_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('senate', 2018): ('G18USSDTES', 'G18USSRROS')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'ne_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'ne_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('senate', 2018): ('G18USSDRAY', 'G18USSRFIS'),
                ('gov', 2018): ('G18GOVDKRI', 'G18GOVRRIC')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'nv_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDCOR', 'G16USSRHEC')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'nv_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('senate', 2018): ('G18USSDROS', 'G18USSRHEL'),
                ('gov', 2018): ('G18GOVDSIS', 'G18GOVRLAX'),
                ('AG', 2018): ('G18ATGDFOR', 'G18ATGRDUN')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'nh_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDHAS', 'G16USSRAYO'),
                ('gov', 2016): ('G16GOVDVAN', 'G16GOVRSUN')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'nh_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('gov', 2018): ('G18GOVDKEL', 'G18GOVRSUN')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'ansolabehere_rodden_2010', 'nj_2010'),
            'precincts': True,
            'county_column': 'COUNTYFP10',
            'elections': {
                ('pres', 2008): ('USP_DV_08', 'USP_RV_08'),
                ('senate', 2008): ('USS_DV_08', 'USS_RV_08')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'ansolabehere_rodden_2010', 'ny_2010'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'nd_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDGLA', 'G16USSRHOE'),
                ('gov', 2016): ('G16GOVDNEL', 'G16GOVRBUR')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'nd_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('senate', 2018): ('G18USSDHEI', 'G18USSRCRA'),
                ('AG', 2018): ('G18ATGDTHO', 'G18ATGRSTE')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'ok_2016'),
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'sc_2016'),
            'precincts': True,
            'county_column': 'COUNTY',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDDIX', 'G16USSRSCO'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'sc_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('gov', 2018): ('G18GOVDSMI', 'G18GOVRMCM'),
                ('AG', 2018): ('G18ATGDANA', 'G18ATGRWIL')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'sd_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDWIL', 'G16USSRTHU'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'sd_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('gov', 2018): ('G18GOVDSUT', 'G18GOVRNOE'),
                ('AG', 2018): ('G18ATGDSEI', 'G18ATGRRAV')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'tn_2016'),
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
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'ut_2018'),
            'precincts': True,
            'county_column': 'CountyID',
            'elections': {
                ('senate', 2018): ('G18USSDWIL', 'G18USSRROM')
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
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'vt_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('AG', 2018): ('G18ATGDDON', 'G18ATGRWIL'),
                ('gov', 2018): ('G18GOVDHAL', 'G18GOVRSCO')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'wa_2016'),
            'precincts': True,
            'county_column': None,
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
                ('senate', 2016): ('G16USSDMUR', 'G16USSRVAN'),
                ('gov', 2016): ('G16GOVDINS', 'G16GOVRBRY'),
                ('AG', 2016): ('G16ATGDFER', 'G16ATGRTRU')
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'wa_2018'),
            'precincts': True,
            'county_column': 'COUNTYCD',
            'elections': {
                ('senate', 2018): ('G18USSDCAN', 'G18USSRHUT')
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
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'wi_2018'),
            'precincts': True,
            'county_column': 'CNTY_FIPS',
            'elections': {
                ('senate', 2018): ('G18USSDBAL', 'G18USSRVUK'),
                ('gov', 2018): ('G18GOVDEVE', 'G18GOVRWAL'),
                ('AG', 2018): ('G18ATGDKAU', 'G18ATGRSCH')
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
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2016', 'wy_2016'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('pres', 2016): ('G16PREDCLI', 'G16PRERTRU'),
            }
        }, {
            'path': os.path.join(constants.PRECINCT_PATH, 'vest_2018', 'wy_2018'),
            'precincts': True,
            'county_column': 'COUNTYFP',
            'elections': {
                ('senate', 2018): ('G18USSDTRA', 'G18USSRBAR'),
                ('gov', 2018): ('G18GOVDTHR', 'G18GOVRGOR')
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