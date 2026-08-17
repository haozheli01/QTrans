============================
Resistivity Calculation
============================

In this tutorial, we will show how to compute the phonon-limited electrical resistivity 
within the momentum relaxation time approximation (MRTA)
by using the ``cal_mode: 'resistivity'`` of **QTrans**.
The relevant theories are briefly introduced in the :ref:`Resistivity <theory_resistivity>` section.

.. warning::
    The following tutorials are designed for instructional purposes. 
    To save computation time, the parameter settings are simplified and the results are not fully converged.

-----------
Bulk fcc Al
-----------

In this tutorial, we use isotropic metal aluminum to introduce the calculation of electrical resistivity.
The bulk fcc aluminum tutorial is located in ``tutorials/Al/resistivity/``. 
All necessary data files are available in ``tutorials/Al/data/``. 
Reference outputs are provided in ``tutorials/Al/resistivity/ref/`` , 
which can be used to compare with your own results to verify that **QTrans** is running correctly.

Enter into the directory, and there would be two yaml files: ``input-dense.yaml`` and 
``input-sparse.yaml`` and the ``ref/`` folder.

.. code-block:: none

    [user]$ cd tutorials/Al/resistivity/ && ls
    input-dense.yaml  input-sparse.yaml ref/

We first use the dense gradient matrix to perform the calculation on a 16x16x16 k/q-point grid.
The smearings for both k-point and q-point integrations are set to 40 meV using ``smeark: 40`` and 
``smearq: 40``. The smearing type is set to the derivative of the Fermi-Dirac function via ``gauss_type: -99``, 
as we are dealing with a metal in this case. The temperature is set to the room temperature with ``temperature: 300.0``. 
Other necessary settings for the resistivity calculation are specified in the ``resistivity`` block.
The parameter ``ziman_nc`` represents the number of carriers per unit cell in the Ziman's resistivity formula. 
It can be set manually. 
If not set, **QTrans** will automatically estimate it based on the number of band crossings at the Fermi level, 
which usually yields a good approximation.
In this case, we use the value estimated by QTrans by leaving it unset.

Launch the **QTrans** calculation with:

.. code-block:: none

    [user]$ mpirun -np 8 QTrans_run --config ./input-dense.yaml > dense_kq16.out

The calculation should take less than 1 min to complete on an 8-CPU machine.
At the end of the run, you should see output like this:

.. code-block:: none

    Ziman's resistance(micro-Ohm*cm) =   8.9537e-01
    Allen's resistance(micro-Ohm*cm) =   2.8265e+00

After the calculation, **QTrans** generates a file named ``a2f.dat``, which contains
the isotropic Eliashberg spectral function 
:math:`\alpha^2 F(\omega)`
and the isotropic Eliashberg transport spectral function
:math:`\alpha_{tr}^2 F(\omega)`, evaluated within the phonon energy range specified by ``omega_range``,
using a step size of ``omega_step``.

Next, we use the sparse gradient matrix to calculate the electrical resistivity.
The same calculation can be performed using:

.. code-block:: none

    [user]$ mpirun -np 8 QTrans_run --config ./input-sparse.yaml > sparse_kq16.out

The calculation should take approximately half the time of the dense case.
At the end of the run, you should see output similar to the following:

.. code-block:: none

    Ziman's resistance(micro-Ohm*cm) =   9.0014e-01
    Allen's resistance(micro-Ohm*cm) =   2.8416e+00

The sparse calculation yields results very close to those of the dense case, but with significantly less computational time.
As shown in our coarse test, Allen's formula produces resistivity values that are in better agreement with the 
experimental result (2.73 :math:`\mu\Omega\cdot\text{cm}`).
However, once the calculations are fully converged, both formulas yield similar results.

We can further plot the Eliashberg spectral functions and extract the electron-phonon coupling strengths
(:math:`\lambda` and :math:`\lambda_{tr}`) using the ``a2f.dat`` file.
The converged results are shown in Figure 1.

.. figure:: ../_static/a2f_Al.png
   :alt: a2f_Al.png
   :width: 600px
   :height: 600px
   :scale: 75 %

   **Figure 1:** The converged Eliashberg and transport spectral functions of Al.  

.. note:: 

    Both Allen's and Ziman's formulas are derived under the assumption of isotropy, and 
    are therefore generally suitable only for isotropic metals.

Because :math:`\rho = \sigma^{-1}`, one can also obtain the resistivity by computing the conductivity using either the IBTE or ERTA method,
as introduced in the :ref:`Conductivity Calculation <tutorial-conductivity>` section. 
This approach is not limited to isotropic metals and is applicable to all metallic systems.

At convergence, the calculated electrical resistivity results comparing with the 
experimental results :cite:`desaiAluminumRho1984` at different temperatures, are shown in Figure 2.

.. figure:: ../_static/resistivity_Al.png
   :alt: resistivity_Al.png
   :width: 600px
   :height: 600px
   :scale: 75 %

   **Figure 2:** The converged resistivities of Al calculated with MRTA and derived from IBTE
   conductivity results using **QTrans**.  
   The experimental data are taken from ref :cite:`desaiAluminumRho1984`.
