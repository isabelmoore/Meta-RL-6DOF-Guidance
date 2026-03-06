# Copyright (c) 2026 Isabel Moore. All rights reserved.
import math
import os


def generate_aircraft_xml(conf, output_dir="jsbsim_data/aircraft/AIM"):
    """Generate JSBSim aircraft XML from YAML config values.

    Args:
        conf: UAVConfig with mass, geometry, aero, and fcs sections.
        output_dir: Directory to write AIM.xml into.

    Returns:
        Path to the generated XML file.
    """
    m = conf.mass
    g = conf.geometry
    a = conf.aero
    f = conf.fcs

    elev_rate = math.radians(f['elevator_rate_max_deg_s'])
    elev_pos = math.radians(f['elevator_pos_max_deg'])
    ail_rate = math.radians(f['aileron_rate_max_deg_s'])
    ail_pos = math.radians(f['aileron_pos_max_deg'])
    rud_rate = math.radians(f['rudder_rate_max_deg_s'])
    rud_pos = math.radians(f['rudder_pos_max_deg'])

    engine_x_m = g['cg_x_in'] * 0.0254 * 1.2

    xml = f"""<?xml version="1.0"?>
<fdm_config name="{conf.name}" version="2.0">

    <fileheader>
        <author>Isabel Moore</author>
        <description>
          {conf.name} — auto-generated from YAML config.
          Weight: {m['weight_lbs']:.1f} lbs, Length: {g['length_in']:.1f} in,
          Diameter: {g['diameter_ft']:.3f} ft
        </description>
    </fileheader>

    <metrics>
        <wingarea unit="FT2"> {g['wingarea_ft2']} </wingarea>
        <wingspan unit="FT"> {g['wingspan_ft']} </wingspan>
        <chord unit="FT"> {g['diameter_ft']} </chord>
        <htailarea unit="FT2"> 0 </htailarea>
        <htailarm unit="FT"> 0 </htailarm>
        <vtailarea unit="FT2"> 0 </vtailarea>
        <vtailarm unit="FT"> 0 </vtailarm>
        <location name="AERORP" unit="IN">
            <x> {g['cg_x_in']} </x>
            <y> 0 </y>
            <z> 0 </z>
        </location>
        <location name="EYEPOINT" unit="IN">
            <x> 0 </x>
            <y> 0 </y>
            <z> 0 </z>
        </location>
        <location name="VRP" unit="IN">
            <x> {g['cg_x_in']} </x>
            <y> 0 </y>
            <z> 0 </z>
        </location>
    </metrics>

    <mass_balance>
        <ixx unit="SLUG*FT2"> {m['ixx']} </ixx>
        <iyy unit="SLUG*FT2"> {m['iyy']} </iyy>
        <izz unit="SLUG*FT2"> {m['izz']} </izz>
        <emptywt unit="LBS"> {m['weight_lbs']} </emptywt>
        <location name="CG" unit="IN">
            <x> {g['cg_x_in']} </x>
            <y> 0 </y>
            <z> 0 </z>
        </location>
    </mass_balance>

    <ground_reactions>
    </ground_reactions>

    <propulsion>
        <engine file="no_thrust_rocket">
            <location unit="M">
                <x> {engine_x_m:.1f} </x>
                <y> 0 </y>
                <z> 0 </z>
            </location>
            <orient unit="DEG">
                <roll> 0.0 </roll>
                <pitch> 0 </pitch>
                <yaw> 0 </yaw>
            </orient>
            <feed>0</feed>
            <thruster file="no_thrust_nozzle">
                <location unit="IN">
                    <x> {engine_x_m:.1f} </x>
                    <y> 0 </y>
                    <z> 0 </z>
                </location>
                <orient unit="DEG">
                    <roll> 0.0 </roll>
                    <pitch> 0.0 </pitch>
                    <yaw> 0.0 </yaw>
                </orient>
            </thruster>
        </engine>
        <tank type="FUEL">
            <location unit="IN">
                <x> {g['cg_x_in']} </x>
                <y> 0 </y>
                <z> 0 </z>
            </location>
            <capacity unit="KG"> 0 </capacity>
            <contents unit="KG"> 0 </contents>
        </tank>
    </propulsion>

    <flight_control name="FCS: {conf.name}">

        <channel name="Pitch">
            <aerosurface_scale name="Elevator Rate Scale">
                <input>fcs/elevator-cmd-norm</input>
                <range>
                    <min> {-elev_rate:.4f} </min>
                    <max>  {elev_rate:.4f} </max>
                </range>
                <output>fcs/elevator-rate-rad_sec</output>
            </aerosurface_scale>
            <integrator name="Elevator Position Integrator">
                <input>fcs/elevator-rate-rad_sec</input>
                <c1> 1.0 </c1>
                <clipto>
                    <min> {-elev_pos:.4f} </min>
                    <max>  {elev_pos:.4f} </max>
                </clipto>
                <output>fcs/elevator-pos-rad</output>
            </integrator>
        </channel>

        <channel name="Roll">
            <aerosurface_scale name="Aileron Rate Scale">
                <input>fcs/aileron-cmd-norm</input>
                <range>
                    <min> {-ail_rate:.5f} </min>
                    <max>  {ail_rate:.5f} </max>
                </range>
                <output>fcs/aileron-rate-rad_sec</output>
            </aerosurface_scale>
            <integrator name="Aileron Position Integrator">
                <input>fcs/aileron-rate-rad_sec</input>
                <c1> 1.0 </c1>
                <clipto>
                    <min> {-ail_pos:.5f} </min>
                    <max>  {ail_pos:.5f} </max>
                </clipto>
                <output>fcs/left-aileron-pos-rad</output>
            </integrator>
            <pure_gain name="Right Aileron Mirror">
                <input>fcs/left-aileron-pos-rad</input>
                <gain> 1.0 </gain>
                <output>fcs/right-aileron-pos-rad</output>
            </pure_gain>
        </channel>

        <channel name="Yaw">
            <aerosurface_scale name="Rudder Rate Scale">
                <input>fcs/rudder-cmd-norm</input>
                <range>
                    <min> {-rud_rate:.4f} </min>
                    <max>  {rud_rate:.4f} </max>
                </range>
                <output>fcs/rudder-rate-rad_sec</output>
            </aerosurface_scale>
            <integrator name="Rudder Position Integrator">
                <input>fcs/rudder-rate-rad_sec</input>
                <c1> 1.0 </c1>
                <clipto>
                    <min> {-rud_pos:.4f} </min>
                    <max>  {rud_pos:.4f} </max>
                </clipto>
                <output>fcs/rudder-pos-rad</output>
            </integrator>
        </channel>

    </flight_control>

    <aerodynamics>

        <axis name="DRAG">
            <function name="aero/coefficient/CA_base">
                <description>Base_axial_drag</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['ca_base']} </value>
                </product>
            </function>
            <function name="aero/coefficient/CA_induced">
                <description>Induced_drag_from_lift</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['ca_induced']} </value>
                    <sum>
                        <abs><property>aero/alpha-rad</property></abs>
                        <abs><property>aero/beta-rad</property></abs>
                    </sum>
                </product>
            </function>
        </axis>

        <axis name="LIFT">
            <function name="aero/coefficient/CN_body_wing">
                <description>Normal_force_body_wing_linear</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['cn_alpha_body_wing']} </value>
                    <property>aero/alpha-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/CN_crossflow">
                <description>Normal_force_body_crossflow</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['cn_crossflow']} </value>
                    <property>aero/alpha-rad</property>
                    <abs><property>aero/alpha-rad</property></abs>
                </product>
            </function>
            <function name="aero/coefficient/CN_tail_alpha">
                <description>Normal_force_tail_due_to_alpha</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['cn_alpha_tail']} </value>
                    <property>aero/alpha-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/CN_tail_elevator">
                <description>Normal_force_tail_due_to_elevator</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['cn_delta_tail']} </value>
                    <property>fcs/elevator-pos-rad</property>
                </product>
            </function>
        </axis>

        <axis name="SIDE">
            <function name="aero/coefficient/CY_body_wing">
                <description>Side_force_body_wing_linear</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['cy_body_wing']} </value>
                    <property>aero/beta-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/CY_crossflow">
                <description>Side_force_body_crossflow</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['cy_crossflow']} </value>
                    <property>aero/beta-rad</property>
                    <abs><property>aero/beta-rad</property></abs>
                </product>
            </function>
            <function name="aero/coefficient/CY_tail_beta">
                <description>Side_force_tail_due_to_beta</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['cy_tail_beta']} </value>
                    <property>aero/beta-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/CY_rudder">
                <description>Side_force_tail_due_to_rudder</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <value> {a['cy_rudder']} </value>
                    <property>fcs/rudder-pos-rad</property>
                </product>
            </function>
        </axis>

        <axis name="PITCH">
            <function name="aero/coefficient/Cm_alpha">
                <description>Pitch_moment_due_to_alpha</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/cbarw-ft</property>
                    <value> {a['cm_alpha']} </value>
                    <property>aero/alpha-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/Cm_elevator">
                <description>Pitch_moment_due_to_elevator</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/cbarw-ft</property>
                    <value> {a['cm_elevator']} </value>
                    <property>fcs/elevator-pos-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/Cm_q">
                <description>Pitch_damping</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/cbarw-ft</property>
                    <property>aero/ci2vel</property>
                    <property>velocities/q-rad_sec</property>
                    <value> {a['cm_q']} </value>
                </product>
            </function>
        </axis>

        <axis name="YAW">
            <function name="aero/coefficient/Cn_beta">
                <description>Yaw_moment_due_to_beta</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/bw-ft</property>
                    <value> {a['cn_beta']} </value>
                    <property>aero/beta-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/Cn_rudder">
                <description>Yaw_moment_due_to_rudder</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/bw-ft</property>
                    <value> {a['cn_rudder']} </value>
                    <property>fcs/rudder-pos-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/Cn_r">
                <description>Yaw_damping</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/bw-ft</property>
                    <property>aero/bi2vel</property>
                    <property>velocities/r-rad_sec</property>
                    <value> {a['cn_r']} </value>
                </product>
            </function>
        </axis>

        <axis name="ROLL">
            <function name="aero/coefficient/Cl_aileron">
                <description>Roll_moment_due_to_aileron</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/bw-ft</property>
                    <property>fcs/left-aileron-pos-rad</property>
                    <value> {a['cl_aileron']} </value>
                </product>
            </function>
            <function name="aero/coefficient/Cl_coupling">
                <description>Roll_coupling_alpha_beta</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/bw-ft</property>
                    <value> {a['cl_coupling']} </value>
                    <property>aero/alpha-rad</property>
                    <property>aero/beta-rad</property>
                </product>
            </function>
            <function name="aero/coefficient/Cl_p">
                <description>Roll_damping</description>
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/bw-ft</property>
                    <property>aero/bi2vel</property>
                    <property>velocities/p-rad_sec</property>
                    <value> {a['cl_p']} </value>
                </product>
            </function>
        </axis>

    </aerodynamics>

</fdm_config>
"""

    os.makedirs(output_dir, exist_ok=True)
    xml_path = os.path.join(output_dir, "AIM.xml")
    with open(xml_path, 'w') as f:
        f.write(xml)
    return xml_path
