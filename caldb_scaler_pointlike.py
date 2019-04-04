import os
import shutil
import glob
import re
import scipy
import astropy.io.fits as pyfits
from matplotlib import pyplot

from scaling_functions import *

# ========================
# ===   Class CalDB   ===
# ========================


class CalDB:
    """
    A class to scale the point-like CTA IRFs, extracted from the ROOT files
    provided by ASWG, stored in the CALDB data base in the FITS format.
    """

    def __init__(self, caldb_name, irf_name, verbose=False):
        """
        Constructor of the class. CALDB data bases will be loaded from the library set by "CALDB" environment variable.

        Parameters
        ----------
        caldb_name: string
            CALDB name to use, e.g. '1dc' or 'prod3b'
        irf_name: string
            IRF name to use, e.g. 'North_z20_50h'.
        verbose: bool, optional
            Defines whether to print additional information during the execution.
        """

        self.caldb_path = os.environ['CALDB']
        self.caldb_name = caldb_name
        self.irf = irf_name
        self.verbose = verbose

        self.am_ok = True

        self._aeff = dict()
        self._psf = dict()
        self._edips = dict()

        self._check_available_irfs()

        self.input_irf_file_name = '{path:s}/{caldb:s}/{irf:s}'.format(path=self.caldb_path,
                                                                                                  caldb=self.caldb_name,
                                                                                                  irf=irf_name)


    def scale_irf(self, config):
        """
        This method performs scaling of the loaded IRF - both PSF and Aeff, if necessary.
        For the collection area two scalings can be applied: (1) vs energy and
        (2) vs off-axis angle. In both cases the scaling function is taken as
        (1 + scale * tanh((x-x0)/dx)). In case (1) the scaling value x is log10(energy).

        Parameters
        ----------
        config: dict
            A dictionary with the scaling settings. Must have following keys defined:
            "general", "aeff", "psf".

            Key "general" must be a dictionary, containing the following:
            caldb: str
                CALDB name, e.g. '1dc' or 'prod3b'.
            irf: str
                IRF name, e.g. 'South_z20_50h'
            output_irf_name: str
                The name of output IRF, say "my_irf".
            output_irf_file_name: str:
                The name of the output IRF file, e.g. 'irf_scaled_version.fits' (the name
                must follow the "irf_*.fits" template, "irf_scaled_version.fits"). The file
                will be put to the main directory of the chosen IRF.

            Keys "aeff" and "psf" must be dictionaries, containing the following:
            "energy_scaling": dict
                Contains setting for the energy scaling (see the structure below).
            "angular_scaling": dict
                Contains setting for the off-center angle scaling (see the structure below).

            In both cases, internally the above dictionaries should contain:
            "err_func_type": str
                The name of the scaling function to use. Accepted values are: "constant",
                "gradient" and "step".

            If err_func_type == "constant":
                scale: float
                    The scale factor. passing 1.0 results in no scaling.

            If err_func_type == "gradient":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                range_min: float
                    The x value (energy or off-center angle), that corresponds to -1 scale.
                range_max: float
                    The x value (energy or off-center angle), that corresponds to +1 scale.

            If err_func_type == "step":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                transition_pos: list
                    The list of x values (energy or off-center angle), at which
                    step-like transitions occur. If scaling the energy dependence,
                    values must be in TeVs, if angular - in degrees.
                transition_widths: list
                    The list of step-like transition widths, that correspond to transition_pos.
                    For energy scaling the widths must be in log10 scale.

        Returns
        -------
        None

        """

        if self.am_ok:
            # Opening the IRF input file
            input_irf_file = pyfits.open(self.input_irf_file_name, 'readonly')

            # Scaling the PSF
            # self._scale_psf(input_irf_file, config['psf'])

            # Scaling the Aeff
            self._scale_aeff(input_irf_file, config['aeff'])

            # Scaling the Edisp
            # self._scale_edisp(input_irf_file, config['edisp'])

            # Getting the new IRF and output file names
            # IRF name
            output_irf_name = config['general']['output_irf_name']
            # Output file name
            output_irf_file_name = config['general']['output_irf_file_name']

            # Figuring out the output path
            output_path = '{path:s}/{caldb:s}_bracketed'.format(
                path=self.caldb_path,
                caldb=self.caldb_name
            )

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Writing the scaled IRF
            input_irf_file.writeto(output_path + "/" + self.irf, overwrite=True)

            # Updating the calibration data base with the new IRF
            # self._append_irf_to_db(output_irf_name, output_irf_file_name)
        else:
            print("ERROR: something's wrong with the CALDB/IRF names. So can not update the data base.")


    def _scale_aeff(self, input_irf_file, config):
        """
        This internal method scales the IRF collection area shape.
        Two scalings can be applied: (1) vs energy and (2) vs off-axis angle. In both cases
        the scaling function is taken as (1 + scale * tanh((x-x0)/dx)). In case (1) the scaling
        is performed in log-energy.

        Parameters
        ----------
        input_irf_file: pyfits.HDUList
            Open pyfits IRF file, which contains the Aeff that should be scaled.

        config: dict
            A dictionary with the scaling settings. Must have following keys defined:
            "energy_scaling": dict
                Contains setting for the energy scaling (see the structure below).
            "angular_scaling": dict
                Contains setting for the off-center angle scaling (see the structure below).

            In both cases, internally the above dictionaries should contain:
            "err_func_type": str
                The name of the scaling function to use. Accepted values are: "constant",
                "gradient" and "step".

            If err_func_type == "constant":
                scale: float
                    The scale factor. passing 1.0 results in no scaling.

            If err_func_type == "gradient":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                range_min: float
                    The x value (energy or off-center angle), that corresponds to -1 scale.
                range_max: float
                    The x value (energy or off-center angle), that corresponds to +1 scale.

            If err_func_type == "step":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                transition_pos: list
                    The list of x values (energy or off-center angle), at which
                    step-like transitions occur. If scaling the energy dependence,
                    values must be in TeVs, if angular - in degrees.
                transition_widths: list
                    The list of step-like transition widths, that correspond to transition_pos.
                    For energy scaling the widths must be in log10 scale.

        Returns
        -------
        None

        """

        # Reading the Aeff parameters
        self._aeff['Elow'] = input_irf_file['SPECRESP'].data['ENERG_LO'][0].copy()
        self._aeff['Ehigh'] = input_irf_file['SPECRESP'].data['ENERG_HI'][0].copy()
        #self._aeff['ThetaLow'] = input_irf_file['Effective area'].data['Theta_lo'][0].copy()
        #self._aeff['ThetaHi'] = input_irf_file['Effective area'].data['Theta_hi'][0].copy()
        self._aeff['Area'] = input_irf_file['SPECRESP'].data['SPECRESP'][0].transpose().copy()
        self._aeff['E'] = scipy.sqrt(self._aeff['Elow'] * self._aeff['Ehigh'])
        # self._aeff['Theta'] = (self._aeff['ThetaLow'] + self._aeff['ThetaHi']) / 2.0

        # Creating the energy-theta mesh grid
        #energy, theta = scipy.meshgrid(self._aeff['E'], self._aeff['Theta'], indexing='ij')
        energy = scipy.meshgrid(self._aeff['E'], indexing='ij')

        # ----------------------------------
        # Scaling the Aeff energy dependence

        # Constant error function
        if config['energy_scaling']['err_func_type'] == "constant":
            self._aeff['Area_new'] = self._aeff['Area'] * config['energy_scaling']['constant']['scale']

        # Gradients error function
        elif config['energy_scaling']['err_func_type'] == "gradient":
            scaling_params = config['energy_scaling']['gradient']
            self._aeff['Area_new'] = self._aeff['Area'] * (
                    1 + scaling_params['scale'] * gradient(scipy.log10(energy),
                                                           scipy.log10(
                                                               scaling_params['range_min']),
                                                           scipy.log10(
                                                               scaling_params['range_max']))
            )

        # Step error function
        elif config['energy_scaling']['err_func_type'] == "step":
            scaling_params = config['energy_scaling']['step']
            break_points = list(zip(scipy.log10(scaling_params['transition_pos']),
                                    scaling_params['transition_widths']))
            self._aeff['Area_new'] = self._aeff['Area'] * (
                    1 + scaling_params['scale'] * step(scipy.log10(energy), break_points)
            )
        else:
            raise ValueError("Aeff energy scaling: unknown scaling function type '{:s}'"
                             .format(config['energy_scaling']['err_func_type']))
        # # ----------------------------------
        #
        # # ------------------------------------------
        # # Scaling the Aeff off-axis angle dependence
        #
        # # Constant error function
        # if config['angular_scaling']['err_func_type'] == "constant":
        #     self._aeff['Area_new'] = self._aeff['Area_new'] * \
        #                              config['angular_scaling']['constant']['scale']
        #
        # # Gradients error function
        # elif config['angular_scaling']['err_func_type'] == "gradient":
        #     scaling_params = config['angular_scaling']['gradient']
        #     self._aeff['Area_new'] = self._aeff['Area_new'] * (
        #             1 + scaling_params['scale'] * gradient(theta,
        #                                                    scaling_params['range_min'],
        #                                                    scaling_params['range_max'])
        #     )
        #
        # # Step error function
        # elif config['angular_scaling']['err_func_type'] == "step":
        #     scaling_params = config['angular_scaling']['step']
        #     break_points = list(zip(scaling_params['transition_pos'],
        #                             scaling_params['transition_widths']))
        #     self._aeff['Area_new'] = self._aeff['Area_new'] * (
        #             1 + scaling_params['scale'] * step(theta, break_points)
        #     )
        # else:
        #     raise ValueError("Aeff angular scaling: unknown scaling function type '{:s}'"
        #                      .format(config['angular_scaling']['err_func_type']))
        # # ------------------------------------------

        from IPython import embed
        embed()

        # Recording the scaled Aeff
        #input_irf_file['SPECRESP'].data['SPECRESP'][0] = self._aeff['Area_new'].transpose()
        input_irf_file['SPECRESP'].data['SPECRESP'] = self._aeff['Area_new'].transpose()


    def _check_available_irfs(self):
        """
        Internal method that checks which CALDB/IRFs are available in the current library.
        Prints an error if the specified CALDB/IRF combination is not found.

        Returns
        -------
        None
        """

        #available_caldb = [path.split('/')[-1] for path in glob.glob(self.caldb_path + '/data/cta/*')]
        available_caldb = [path.split('/')[-1] for path in glob.glob(self.caldb_path + '/*')]

        if self.verbose:
            print('-- Available CALDBs -- ')
            print("  {}".format(available_caldb))

        for caldb in available_caldb:
            #available_irf = [path.split('/')[-1] for path in glob.glob(self.caldb_path + '/data/cta/' + caldb + '/bcf/*')]
            available_irf = [path.split('/')[-1] for path in glob.glob(self.caldb_path + '/' + caldb + '/*')]

            if self.verbose:
                print("-- Available IRFs for '{:s}' --".format(caldb))
                print("  {}".format(available_irf))

        if self.caldb_name not in available_caldb:
            print("ERROR: provided CALDB name '{:s}' is not found!".format(self.caldb_name))
            print('Available options are:')
            print("  {}".format(available_caldb))
            self.am_ok = False

        if self.irf not in available_irf:
            print("ERROR: provided IRF name '{:s}' is not found!".format(self.irf))
            print('Available options are:')
            print("  {}".format(available_irf))
            self.am_ok = False