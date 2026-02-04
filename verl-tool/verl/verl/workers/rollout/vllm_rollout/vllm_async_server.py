# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import pickle
from typing import Any, Callable, Optional

import ray
import zmq
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase

logger = logging.getLogger(__file__)


def _get_model_runner_workers(vllm_config, init_ray: bool = True):
    assert vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

    fields = vllm_config.instance_id.split(":")
    assert len(fields) == 4, (
        f"instance_id: {vllm_config.instance_id} must be in the format of "
        f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
    )
    namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

    # Make sure subprocess in same namespace as parent actor.
    # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
    if init_ray:
        ray.init(namespace=namespace)
    actor_names = [
        actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")
    ]

    vllm_tp_size = vllm_config.parallel_config.tensor_parallel_size
    assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
        f"instance_id: {vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: "
        f"{vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."
    )

    def get_pg_index_and_local_rank(actor_name) -> tuple[int, int]:
        fields = actor_name.split(":")
        assert len(fields) == 2, f"invalid actor name: {actor_name}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
        return pg_index, local_rank

    # sort actor names by pg_index and local_rank
    actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
    actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
    workers: list[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
    print(f"instance_id: {vllm_config.instance_id} initializes with external actors: {actor_names}")

    return workers


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        self.workers = _get_model_runner_workers(vllm_config=self.vllm_config, init_ray=True)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} initializes finished.")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        # ~3ms overhead per schedule step due to SchedulerOutput/ModelRunnerOutput serialization/deserialization.
        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers]
        )
        return outputs

    def check_health(self):
        return


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))
        return outputs

    def check_health(self):
        return

# 一定要改回来这里，测试之后
# @ray.remote(num_cpus=1, num_gpus=1)
@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        Args:
            config: DictConfig.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config.actor_rollout_ref
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        self.max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            repetition_penalty=1.0,
            max_new_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "zeromq")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None   # ✅ 单机模式

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=config.free_cache_engine,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=self.max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            # continue_final_message=True,
            # add_generation_prompt=False,
            chat_template_content_format="auto",
            enable_auto_tools=config.multi_turn.tool_config_path is not None,
            tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
        )

    def _create_engine_config(self, engine_args: AsyncEngineArgs):
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"

        # VERL_VLLM_ZMQ_ADDRESSES
        if engine_args.distributed_executor_backend == ExternalZeroMQDistributedExecutor:
            workers = _get_model_runner_workers(vllm_config=vllm_config, init_ray=False)
            zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in workers])
            print(f"VERL_VLLM_ZMQ_ADDRESSES: {zmq_addresses}")
            os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        return vllm_config

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
        max_tokens = self.max_model_len - len(prompt_ids)
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        return final_res.outputs[0].token_ids

    async def wake_up(self):
        if self.config.rollout.free_cache_engine:
            await self.engine.wake_up()

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        if self.config.rollout.free_cache_engine:
            await self.engine.sleep()
if __name__ == "__main__":
    import os
    os.environ["VLLM_USE_V1"] = "1"   # ✅ 启用 vLLM V1 异步引擎
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VERL_VLLM_DISTRIBUTED_BACKEND"] = "none"
    # 确保 vLLM 不再自动修改 multiprocessing 启动方式
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch
    print(f"[DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[DEBUG] torch.cuda.device_count() = {torch.cuda.device_count()}")
    import asyncio
    from omegaconf import OmegaConf
    from unittest.mock import MagicMock, patch
    import ray

    print("🚀 Starting AsyncvLLMServer self-test with real model path...")

    # 初始化 Ray 环境
    ray.init(ignore_reinit_error=True, namespace="testns", num_gpus=1)

    # ===========================
    # 配置部分（已替换模型路径）
    # ===========================
    config = OmegaConf.create({
        "actor_rollout_ref": {
            "model": {
                # 👇 这里是真实模型路径
                "path": "/data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "trust_remote_code": True,  # 对 deepseek 模型建议启用
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "max_model_len": 8192,
                "response_length": 256,
                "prompt_length": 512,
                "dtype": "float16",
                "free_cache_engine": False,
                "gpu_memory_utilization": 0.9,
                "disable_log_stats": True,
                "enable_chunked_prefill": False,
                "enforce_eager": False,
                "multi_turn": {"tool_config_path": None, "format": "hermes"},
            },
        }
    })

    # =======================================================
    # Mock 掉 vLLM 的大部分函数，防止真实加载模型
    # =======================================================
    with patch("verl.utils.fs.copy_to_local", return_value="/data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"), \
         patch("vllm.v1.engine.async_llm.AsyncLLM.from_vllm_config") as mock_from_cfg, \
         patch("vllm.entrypoints.openai.serving_models.OpenAIServingModels"), \
         patch("vllm.entrypoints.openai.serving_chat.OpenAIServingChat"), \
         patch("vllm.entrypoints.logger.RequestLogger"):

        fake_engine = MagicMock()
        fake_engine.model_config = MagicMock()
        # 模拟 engine.generate() 异步生成结果
        async def fake_generator(*args, **kwargs):
            fake_output = MagicMock()
            fake_output.outputs = [MagicMock(token_ids=[101, 102, 103])]
            yield fake_output

        fake_engine.generate.return_value = fake_generator()
        mock_from_cfg.return_value = fake_engine

        # ✅ 正确的 Actor 创建方式
        server_ref = AsyncvLLMServer.remote(config=config, vllm_dp_size=1, vllm_dp_rank=0, wg_prefix="wg")

        async def run_test():
            # 初始化引擎
            ray.get(server_ref.init_engine.remote())
            print("✅ Engine initialized successfully (mocked).")

            # 运行一次生成任务
            result = ray.get(server_ref.generate.remote([151646, 16141, 279, 2661, 3405, 13, 1446, 1969, 6786, 32711, 4766, 220, 151648, 323, 220, 151649, 1156, 1449, 882, 498, 633, 501, 1995, 13, 1416, 498, 1477, 498, 6853, 1045, 6540, 476, 63684, 374, 2567, 11, 498, 646, 1618, 264, 10161, 4712, 553, 366, 70705, 29, 3239, 690, 70705, 29, 323, 432, 686, 470, 279, 11223, 1995, 1948, 366, 2322, 29, 323, 690, 2322, 14276, 1446, 646, 2548, 438, 1657, 3039, 438, 697, 1366, 13, 1416, 498, 1477, 902, 4623, 9250, 6540, 4362, 11, 498, 646, 5961, 3410, 279, 4226, 4766, 366, 9217, 29, 323, 690, 9217, 8066, 2041, 11682, 44443, 13, 151644, 32, 3669, 3910, 64485, 458, 5461, 315, 220, 17, 15, 4948, 323, 9104, 3669, 9709, 1449, 1899, 13, 20517, 279, 2790, 1372, 315, 9709, 279, 1378, 13037, 4652, 3786, 304, 7400, 421, 1052, 525, 220, 17, 23, 2849, 304, 279, 2254, 5209, 2874, 3019, 553, 3019, 11, 323, 2182, 697, 1590, 4226, 2878, 1124, 79075, 46391, 151645, 151648, 198], {"temperature": 0.8}, "req_01"))
            print("✅ Generation result:", result)

        asyncio.run(run_test())

    print("🎉 AsyncvLLMServer Ray self-test finished.")