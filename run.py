import copy
import random
import numpy as np
import yaml
import os
from rich import print

from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.driver_agent.reflectionAgent import ReflectionAgent
from dilu.driver_agent.safetyShield import SafetyShield, ShieldConfig


test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]


def setup_env(config):
    if config['OPENAI_API_TYPE'] == 'azure':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_VERSION"] = config['AZURE_API_VERSION']
        os.environ["OPENAI_API_BASE"] = config['AZURE_API_BASE']
        os.environ["OPENAI_API_KEY"] = config['AZURE_API_KEY']
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = config['AZURE_CHAT_DEPLOY_NAME']
        os.environ["AZURE_EMBED_DEPLOY_NAME"] = config['AZURE_EMBED_DEPLOY_NAME']

    elif config['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
        os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']

    elif config['OPENAI_API_TYPE'] == 'deepseek':
        # IMPORTANT:
        # DeepSeek is OpenAI-compatible, but OpenAI SDK does NOT accept api_type="deepseek".
        # So we pretend to be openai, and only change base_url + key.
        os.environ["OPENAI_API_TYPE"] = 'openai'  # ✅ critical
        os.environ["OPENAI_API_KEY"] = config['DEEPSEEK_API_KEY']
        os.environ["OPENAI_API_BASE"] = config.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        os.environ["OPENAI_CHAT_MODEL"] = config.get('DEEPSEEK_CHAT_MODEL', 'deepseek-chat')

    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be azure / openai / deepseek")

    # memory embedding backend config (used by DrivingMemory)
    if 'EMBEDDING_BACKEND' in config:
        os.environ['EMBEDDING_BACKEND'] = str(config['EMBEDDING_BACKEND'])
    if 'HF_EMBED_MODEL' in config:
        os.environ['HF_EMBED_MODEL'] = str(config['HF_EMBED_MODEL'])

    # environment setting
    env_config = {
        'highway-v0': {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["vehicle_count"],
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },
            "lanes_count": 4,
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": True,
            "render_agent": True,
            "scaling": 5,
            "initial_lane_id": None,
            "ego_spacing": 4,
        }
    }

    return env_config


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    print("[magenta]episodes_num from config =", config.get("episodes_num"))

    env_config = setup_env(config)

    REFLECTION = config["reflection_module"]
    memory_path = config["memory_path"]
    few_shot_num = config["few_shot_num"]

    # base result folder from config
    base_result_folder = config["result_folder"]

    # create a unique subfolder for this run
    run_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
    result_folder = os.path.join(base_result_folder, run_name)

    os.makedirs(result_folder, exist_ok=True)

    print(f"[green]Results will be saved to:[/green] {result_folder}")

    with open(os.path.join(result_folder, 'log.txt'), 'a') as f:
        f.write("===== New Run =====\n")
        f.write(f"episodes_num: {config.get('episodes_num')}\n")
        f.write(f"enable_safety_shield: {config.get('enable_safety_shield')}\n")
        f.write(f"enable_diverse_memory: {config.get('enable_diverse_memory')}\n")
        f.write(f"enable_structured_reflection: {config.get('enable_structured_reflection')}\n\n")


    with open(os.path.join(result_folder, 'log.txt'), 'w') as f:
        f.write("memory_path {} | result_folder {} | few_shot_num: {} | lanes_count: {} \n".format(
            memory_path, result_folder, few_shot_num, env_config['highway-v0']['lanes_count']))

    # ---- Base memory DB (for few-shot) ----
    agent_memory = DrivingMemory(db_path=memory_path)

    # ---- Shield memory DB (near-miss / blocked actions) ----
    shield_memory = DrivingMemory(db_path=memory_path + "_shield")

    # ---- Updated memory DB (optional: reflection-updated) ----
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=memory_path + "_updated")
        updated_memory.combineMemory(agent_memory)

    episode = 0
    while episode < config["episodes_num"]:
        envType = 'highway-v0'
        env = gym.make(envType, render_mode="rgb_array")
        env.configure(env_config[envType])

        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)

        seed = random.choice(test_list_seed)
        obs, info = env.reset(seed=seed)
        env.render()

        # scenario and driver agent setting
        database_path = os.path.join(result_folder, result_prefix + ".db")
        sce = EnvScenario(env, envType, seed, database_path)
        DA = DriverAgent(sce, verbose=True)

        # ---- Safety Shield (C) ----
        shield = SafetyShield(
            cfg=ShieldConfig(
                min_front_gap_m=config.get("shield_min_front_gap_m", 12.0),
                min_back_gap_m=config.get("shield_min_back_gap_m", 8.0),
                min_front_gap_lc_m=config.get("shield_min_front_gap_lc_m", 12.0),
                ttc_front_s=config.get("shield_ttc_front_s", 2.5),
                ttc_back_s=config.get("shield_ttc_back_s", 1.5),
                fallback_action=config.get("shield_fallback_action", 4),
            ),
            verbose=True
        )

        # ---- Reflection Agent (D) ----
        if REFLECTION:
            RA = ReflectionAgent(verbose=True)

        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1
        already_decision_steps = 0

        try:
            for i in range(0, config["simulation_duration"]):
                obs = np.array(obs, dtype=float)

                # ---- E: Diverse few-shot retrieval (MMR) ----
                print("[cyan]Retreive similar memories...[/cyan]")
                if few_shot_num > 0:
                    fewshot_results = agent_memory.retriveMemory(
                        sce, i, few_shot_num,
                        diverse=config.get("enable_diverse_memory", True),
                        candidate_pool_k=config.get("memory_candidate_pool_k", 20),
                        mmr_lambda=config.get("mmr_lambda", 0.7),
                    )
                else:
                    fewshot_results = []

                fewshot_messages = []
                fewshot_answers = []
                fewshot_actions = []

                for fewshot_result in fewshot_results:
                    fewshot_messages.append(fewshot_result["human_question"])
                    fewshot_answers.append(fewshot_result["LLM_response"])
                    fewshot_actions.append(fewshot_result["action"])

                if few_shot_num == 0:
                    print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                else:
                    print("[green4]Successfully find[/green4]", len(fewshot_actions), "[green4]similar memories![/green4]")

                # ---- Describe scenario & decide ----
                sce_descrip = sce.describe(i)
                avail_action = sce.availableActionsDescription()
                print('[cyan]Scenario description: [/cyan]\n', sce_descrip)

                action, response, human_question, fewshot_answer = DA.few_shot_decision(
                    scenario_description=sce_descrip,
                    available_actions=avail_action,
                    previous_decisions=action,
                    fewshot_messages=fewshot_messages,
                    driving_intensions="Drive safely and avoid collisons",
                    fewshot_answers=fewshot_answers,
                )

                # log docs for reflection
                docs.append({
                    "sce_descrip": sce_descrip,
                    "human_question": human_question,
                    "response": response,
                    "action": action,
                    "sce": copy.deepcopy(sce),
                })

                # ---- C: Safety Shield + write near-miss to shield_memory ----
                if config.get("enable_safety_shield", True):
                    safe_action, shield_info = shield.enforce(sce, action)

                    if shield_info.get("blocked", False):
                        print(f"[yellow][Shield Blocked][/yellow] {shield_info}")

                        if config.get("enable_shield_memory", True):
                            # avoid overwrite due to "$contains" update logic
                            shield_tag = f"\n[SHIELD_EVENT frame={i} reason={shield_info.get('reason')}]"
                            sce_descrip_for_store = sce_descrip + shield_tag

                            event_note = (
                                f"\n[Shield] proposed={shield_info.get('proposed_action')} "
                                f"final={shield_info.get('final_action')} reason={shield_info.get('reason')} "
                                f"front_gap={shield_info.get('cur_front_gap', None)} "
                                f"front_ttc={shield_info.get('cur_front_ttc', None)}"
                            )

                            shield_memory.addMemory(
                                sce_descrip_for_store,
                                human_question + event_note,
                                response + event_note,
                                safe_action,
                                sce,
                                comments="shield-block"
                            )

                    action = safe_action

                # ---- step env ----
                obs, reward, done, info, _ = env.step(action)
                already_decision_steps += 1

                env.render()
                sce.promptsCommit(i, None, done, human_question, fewshot_answer, response)
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

                print("--------------------")

                if done:
                    print("[red]Simulation crash after running steps: [/red] ", i)
                    collision_frame = i
                    break

        finally:
            with open(os.path.join(result_folder, 'log.txt'), 'a') as f:
                f.write("Simulation {} | Seed {} | Steps: {} | File prefix: {} \n".format(
                    episode, seed, already_decision_steps, result_prefix))

            # ---- D: Reflection + Structured Rule 저장 ----
            if REFLECTION:
                print("[yellow]Now running reflection agent...[/yellow]")

                # if collision happened, reflect on a non-decelerate action near the crash
                if collision_frame != -1:
                    for j in range(collision_frame, -1, -1):
                        if docs[j]["action"] != 4:  # not decelerate
                            if config.get("enable_structured_reflection", True):
                                corrected_text, rule = RA.reflection_with_rule(
                                    docs[j]["human_question"], docs[j]["response"]
                                )
                                extra_meta = {"rule_json": rule} if rule else {}
                                corrected_response = corrected_text
                            else:
                                corrected_response = RA.reflection(
                                    docs[j]["human_question"], docs[j]["response"]
                                )
                                extra_meta = {}

                            choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): [/yellow]").strip().upper()
                            if choice == 'Y':
                                updated_memory.addMemory(
                                    docs[j]["sce_descrip"],
                                    docs[j]["human_question"],
                                    corrected_response,
                                    docs[j]["action"],
                                    docs[j]["sce"],
                                    comments="mistake-correction",
                                    extra_meta=extra_meta
                                )
                                print("[green]Successfully added a new memory item.[/green] Now the database has ",
                                      len(updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']),
                                      " items.")
                            else:
                                print("[blue]Ignore this new memory item[/blue]")
                            break

                else:
                    # no collision: optionally add some direct experiences
                    print("[yellow]Do you want to add[/yellow]", len(docs) // 5, "[yellow]new memory item to update memory module?[/yellow]", end="")
                    choice = input("(Y/N): ").strip().upper()
                    if choice == 'Y':
                        cnt = 0
                        for j in range(0, len(docs)):
                            if j % 5 == 1:
                                updated_memory.addMemory(
                                    docs[j]["sce_descrip"],
                                    docs[j]["human_question"],
                                    docs[j]["response"],
                                    docs[j]["action"],
                                    docs[j]["sce"],
                                    comments="no-mistake-direct",
                                    extra_meta={}
                                )
                                cnt += 1
                        print("[green]Successfully added[/green]", cnt, "[green]new memory items.[/green] Now the database has ",
                              len(updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']),
                              " items.")
                    else:
                        print("[blue]Ignore these new memory items[/blue]")

            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close()
