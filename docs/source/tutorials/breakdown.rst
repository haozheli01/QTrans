===============================
Intrinsic Dielectric Breakdown
===============================

In this tutorial, we will show how to compute the
intrinsic breakdown field strength using **QTrans**
with :ref:`cal_mode <cal_cal_mode>` set to be ``breakdown``.
We use typical semiconductor silicon (Si) for example.
The relevant theories are briefly introduced in the
:ref:`Theory/Intrinsic Dielectric Breakdown <theory-breakdown>` section.

.. warning::
    The following tutorials are designed for instructional purposes. 
    To save computation time, the parameter settings are simplified and the results are not fully converged.

---------
Bulk Si
---------

The bulk Si tutorial is located in
``tutorials/Si/breakdown/``. 
All necessary data files are available in ``tutorials/Si/data``. 
Reference outputs are provided in ``tutorials/Si/breakdown/ref/`` , 
which can be used to compare with your own results to verify that **QTrans** is running correctly.

Once **QTrans** is installed in your Conda environment, you can easily run the following tutorials.

The first step is enter into the directory, and there would be two yaml files: ``input-dense.yaml`` and 
``input-sparse.yaml``, and the ``ref/`` folder.
The former yaml file uses the dense gradient matrix for higher accuracy, while the latter uses a much smaller sparse 
gradient matrix for several times faster computation, with negligible accuracy loss.
Here, by setting :ref:`read_sparse_grad_mat <cal_read_sparse_grad_mat>` to ``True``,
we use the sparse gradient matrix to reduce computational cost.

.. code-block:: none

    [user]$ cd tutorials/Si/breakdown/
    [user]$ ls
    input-dense.yaml  input-sparse.yaml ref/

Activate the Conda environment, then we can start calculating the intrinsic breakdown field strength for Si:

.. code-block:: none

    [user]$ conda activate QTrans

.. _fig-band-Si-breakdown:

.. figure:: ../_static/band_Si_breakdown.png
   :alt: band_Si_breakdown.png
   :width: 600px
   :height: 600px
   :scale: 75 %

   **Figure 1:**  The calculated band structures of silicon.  
   The red dashed line denotes the energy level of :math:`\mathrm{CBM} + E_g` .  
   The yellow region indicates the energy window considered in the calculation.

Here, we perform a **QTrans** calculation to calculate the intrinsic breakdown field strength for Si.
We use :math:`16 \times 16 \times 16` **k**-point and **q**-point meshes.

You can first check the input flags in ``input-sparse.yaml`` and make sure you understand the meaning of each tag.
We consider the first to fourth conduction bands by setting 
:ref:`bands_indices <cal_bands_indices>` to ``'5-8'``. 
Since the experimental band gap of silicon is approximately 1.17 eV, 
we set :ref:`fbd_erange <cal_fbd_erange>` to ``[0.0, 2.0, 0.01]`` 
to include electron states within the energy range 
:math:`\left[\text{CBM}, \text{CBM} + 2.0\;\text{eV}\right]`, 
where CBM denotes the conduction band minimum. 
This energy window is highlighted in yellow in :ref:`Figure 1 <fig-band-Si-breakdown>`. 
The parameter :ref:`fbd_effective_mass <cal_fbd_effective_mass>` is set to ``0.96``, 
corresponding to the electron effective mass at the CBM, 
and :ref:`fbd_exp_bandgap <cal_fbd_exp_bandgap>` is set to ``1.17``. 
The temperature is set to room temperature via :ref:`temperature <cal_temperature>` with a value of ``300``.

You can launch the **QTrans** calculation with:

.. code-block:: none

    [user]$ mpirun -np 8 QTrans_run --config ./input-sparse.yaml > sparse_kq16_breakdown.out

The calculation should take about 2 minutes to complete on an 8-CPU machine.
At the end of the run, you should see output like this:

.. code-block:: none

    Breakdown field strength (MV/m): 120.54662074968543

The program will generate a file named ``fbd_rate_nk.dat``,
which contains scattering rates and energy loss rates dependent on electron states,
and can be used to restart the calculation.

.. warning::

    The restart running is not yet implemented.

In addition, the program generates a file named ``fbd_rate_ene.dat``, with the following structure:

.. code-block:: none

    energy       DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate           DOS          scatt-rate       eloss-rate    
    eV         1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps             1/eV             1/ps            eV/ps      
    ----     25.00 meV        25.00 meV        25.00 meV        50.00 meV        50.00 meV        50.00 meV        75.00 meV        75.00 meV        75.00 meV        100.00meV        100.00meV        100.00meV        125.00meV        125.00meV        125.00meV        150.00meV        150.00meV        150.00meV        175.00meV        175.00meV        175.00meV        200.00meV        200.00meV        200.00meV        225.00meV        225.00meV        225.00meV        250.00meV        250.00meV        250.00meV    
    0.0000   8.00728272e-02   3.03104916e+00   1.37797626e-02   5.54693678e-02   3.53753135e+00   5.64162749e-02   4.18150255e-02   4.38738104e+00   6.66999650e-02   4.00516675e-02   9.06770081e+00   8.11326999e-02   4.52623481e-02   1.48236330e+01   1.14058669e-01   5.17069468e-02   1.88713281e+01   1.58509020e-01   5.76037314e-02   2.16360429e+01   2.10895481e-01   6.28547371e-02   2.38228427e+01   2.70844215e-01   6.76708187e-02   2.58671655e+01   3.37588160e-01   7.22604115e-02   2.80285018e+01   4.10724062e-01
    0.0100   8.85964735e-02   3.44814835e+00   4.90087666e-02   5.94034869e-02   3.67600338e+00   6.77731764e-02   4.42073523e-02   4.95382690e+00   7.20606607e-02   4.40202459e-02   1.05293974e+01   8.75782429e-02   5.03012853e-02   1.61418891e+01   1.22968695e-01   5.69485059e-02   1.98180827e+01   1.68876789e-01   6.27018703e-02   2.23363681e+01   2.22603388e-01   6.77324623e-02   2.43942101e+01   2.83884168e-01   7.23411681e-02   2.63874318e+01   3.51870623e-01   7.67625179e-02   2.85476493e+01   4.26202911e-01
    0.0200   8.89956161e-02   3.98483341e+00   9.43382253e-02   5.96615466e-02   3.82591827e+00   7.93115087e-02   4.58883980e-02   5.83380424e+00   7.75469874e-02   4.83621434e-02   1.22081586e+01   9.48824693e-02   5.57838004e-02   1.74747896e+01   1.32612584e-01   6.25209500e-02   2.07441180e+01   1.79873168e-01   6.80437969e-02   2.30219700e+01   2.34944088e-01   7.28010223e-02   2.49610593e+01   2.97553488e-01   7.71708147e-02   2.69119435e+01   3.66767520e-01   8.14049640e-02   2.90776280e+01   4.42282021e-01
    0.0300   8.16337144e-02   4.42405284e+00   1.31435548e-01   5.62443211e-02   4.00339954e+00   9.06312663e-02   4.71559117e-02   7.17038565e+00   8.31754202e-02   5.32350750e-02   1.40636360e+01   1.03109176e-01   6.17382926e-02   1.87972363e+01   1.42981348e-01   6.84164191e-02   2.16428116e+01   1.91518897e-01   7.36169837e-02   2.36917518e+01   2.47949235e-01   7.80499547e-02   2.55239203e+01   3.11880268e-01   8.21521708e-02   2.74417712e+01   3.82302806e-01   8.61825874e-02   2.96195947e+01   4.58981785e-01
    0.0400   6.34881367e-02   4.66886008e+00   1.52112440e-01   4.98809859e-02   4.26835027e+00   1.01238522e-01   4.85186578e-02   9.10526001e+00   8.89878453e-02   5.88111641e-02   1.60294698e+01   1.12277689e-01   6.81827051e-02   2.00852247e+01   1.54062811e-01   7.46206416e-02   2.25087540e+01   2.03840270e-01   7.94058355e-02   2.43451370e+01   2.61653415e-01   8.34672984e-02   2.60835781e+01   3.26893657e-01   8.72768901e-02   2.79781322e+01   3.98500942e-01   9.10898214e-02   3.01747814e+01   4.76322687e-01

The first two rows contain the column titles and their corresponding units.
The third row lists the smearing parameters used for each column when integrating over electron states.
These smearing parameters range from one to ten times the value of :ref:`smeark <cal_smeark>`,
which helps verify the convergence of the integration.
The intrinsic breakdown field strength reported in the standard output file
is calculated using the smearing parameter equal to :ref:`smeark <cal_smeark>`.
