var Ot=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function Ug(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}function wr(e){if(e.__esModule)return e;var t=e.default;if(typeof t=="function"){var n=function s(){return this instanceof s?Reflect.construct(t,arguments,this.constructor):t.apply(this,arguments)};n.prototype=t.prototype}else n={};return Object.defineProperty(n,"__esModule",{value:!0}),Object.keys(e).forEach(function(s){var r=Object.getOwnPropertyDescriptor(e,s);Object.defineProperty(n,s,r.get?r:{enumerable:!0,get:function(){return e[s]}})}),n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yr=1e-7,$r=1e-4;class Gg{constructor(t,n){this.backend=t,this.dataMover=n,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,n){this.dataIdsCount++,this.data.set(t,n)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class kr{refCount(t){return nt("refCount")}incRef(t){return nt("incRef")}timerAvailable(){return!0}time(t){return nt("time")}read(t){return nt("read")}readSync(t){return nt("readSync")}readToGPU(t,n){return nt("readToGPU")}numDataIds(){return nt("numDataIds")}disposeData(t,n){return nt("disposeData")}write(t,n,s){return nt("write")}move(t,n,s,r,o){return nt("move")}createTensorFromGPUData(t,n,s){return nt("createTensorFromGPUData")}memory(){return nt("memory")}floatPrecision(){return nt("floatPrecision")}epsilon(){return this.floatPrecision()===32?yr:$r}dispose(){return nt("dispose")}}function nt(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zg(e,t,n){const s=e[t];e[t]=e[n],e[n]=s}function qg(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function p(e,t){if(!e)throw new Error(typeof t=="string"?t:t())}function ut(e,t,n=""){p(Lt(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function ee(e){p(e!=null,()=>"The input to the tensor constructor must be a non-null value.")}function H(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function Hg(e){return e.length===0}function xr(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==null&&t[n]!==null&&e[n]!==t[n])return!1;return!0}function Lt(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function ue(e){return e%1===0}function we(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function Vg(e,t){let n=1,s=-1;for(let o=0;o<e.length;++o)if(e[o]>=0)n*=e[o];else if(e[o]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${o}`);s=o}else if(e[o]<0)throw Error(`Shapes can not be < 0. Found ${e[o]} at dim ${o}`);if(s===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const r=e.slice();return r[s]=t/n,r}function Fe(e,t){const n=t.length;return e=e==null?t.map((s,r)=>r):[].concat(e),p(e.every(s=>s>=-n&&s<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),p(e.every(s=>ue(s)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(s=>s<0?n+s:s)}function Er(e,t){const n=[],s=[],r=t!=null&&Array.isArray(t)&&t.length===0,o=t==null||r?null:Fe(t,e).sort();let a=0;for(let i=0;i<e.length;++i){if(o!=null){if(o[a]===i&&e[i]!==1)throw new Error(`Can't squeeze axis ${i} since its dim '${e[i]}' is not 1`);(o[a]==null||o[a]>i)&&e[i]===1&&(n.push(e[i]),s.push(i)),o[a]<=i&&a++}e[i]!==1&&(n.push(e[i]),s.push(i))}return{newShape:n,keptDims:s}}function vr(e,t){return bs(e,t)}function bs(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else if(e==="string")n=new Array(t);else throw new Error(`Unknown data type ${e}`);return n}function Sr(e,t){for(let n=0;n<e.length;n++){const s=e[n];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${t} being uploaded contains ${s}.`)}}function Ir(e){return e==="bool"||e==="complex64"||e==="float32"||e==="int32"||e==="string"}function jg(e,t){return!(t==="complex64"||t==="float32"&&e!=="complex64"||t==="int32"&&e!=="float32"&&e!=="complex64"||t==="bool"&&e==="bool")}function Xe(e){if(e==="float32"||e==="int32")return 4;if(e==="complex64")return 8;if(e==="bool")return 1;throw new Error(`Unknown dtype ${e}`)}function Tr(e){if(e==null)return 0;let t=0;return e.forEach(n=>t+=n.length),t}function gn(e){return typeof e=="string"||e instanceof String}function Dr(e){return typeof e=="boolean"}function Ar(e){return typeof e=="number"}function Re(e){return Array.isArray(e)?Re(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":Ar(e)?"float32":gn(e)?"string":Dr(e)?"bool":"float32"}function Je(e){return!!(e&&e.constructor&&e.call&&e.apply)}function ge(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let s=t-3;s>=0;--s)n[s]=n[s+1]*e[s+1];return n}function ws(e,t,n,s=!1){const r=new Array;if(t.length===1){const o=t[0]*(s?2:1);for(let a=0;a<o;a++)r[a]=n[e+a]}else{const o=t[0],a=t.slice(1),i=a.reduce((c,l)=>c*l)*(s?2:1);for(let c=0;c<o;c++)r[c]=ws(e+c*i,a,n,s)}return r}function zn(e,t,n=!1){if(e.length===0)return t[0];const s=e.reduce((r,o)=>r*o)*(n?2:1);if(s===0)return[];if(s!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return ws(0,e,t,n)}function Xg(e,t){if(Array.isArray(e))return e;if(t==="float32")return e instanceof Float32Array?e:new Float32Array(e);if(t==="int32")return e instanceof Int32Array?e:new Int32Array(e);if(t==="bool"||t==="string")return Uint8Array.from(new Int32Array(e));throw new Error(`Unknown dtype ${t}`)}function ys(e,t){const n=mn(e,t);for(let s=0;s<n.length;s++)n[s]=1;return n}function mn(e,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool")return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function ot(e){e.forEach(t=>{p(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function Jg(e,t,n){if(t===0)return 0;if(t===1)return e[0];let s=e[e.length-1];for(let r=0;r<e.length-1;++r)s+=n[r]*e[r];return s}function Yg(e,t,n){if(t===0)return[];if(t===1)return[e];const s=new Array(t);for(let r=0;r<s.length-1;++r)s[r]=Math.floor(e/n[r]),e-=s[r]*n[r];return s[s.length-1]=e,s}function bn(e){return e&&e.then&&typeof e.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qn="tfjsflags";class _r{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Nr,this.populateURLFlags()}setPlatform(t,n){this.platform!=null&&(R().getBool("IS_TEST")||R().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=n}registerFlag(t,n,s){if(this.flagRegistry[t]={evaluationFn:n,setHook:s},this.urlFlags[t]!=null){const r=this.urlFlags[t];R().getBool("IS_TEST")||R().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${r}.`),this.set(t,r)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const n=this.evaluateFlag(t);if(bn(n))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=n,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getString(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,n){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=n,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(n)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);qn in t&&t[qn].split(",").forEach(s=>{const[r,o]=s.split(":");this.urlFlags[r]=Cr(r,o)})}}function Nr(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...s)=>(Mr(t,s[0],s[1]),s.join("="))),t}function Mr(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}function Cr(e,t){const n=t.toLowerCase();return n==="true"||n==="false"?n==="true":`${+n}`===n?+n:t}function R(){return $s}let $s=null;function Br(e){$s=e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ke;function ks(){if(Ke==null){let e;if(typeof window<"u")e=window;else if(typeof global<"u")e=global;else if(typeof process<"u")e=process;else if(typeof self<"u")e=self;else throw new Error("Could not find a global object");Ke=e}return Ke}function Fr(){const e=ks();return e._tfGlobals==null&&(e._tfGlobals=new Map),e._tfGlobals}function wn(e,t){const n=Fr();if(n.has(e))return n.get(e);{const s=t();return n.set(e,s),n.get(e)}}const Rr="Abs",Pr="Acos",Or="Acosh",xs="Add",Lr="AddN",Wr="All",Kr="Any",Ur="ArgMax",Gr="ArgMin",zr="Asin",qr="Asinh",Hr="Atan",Vr="Atanh",jr="Atan2",Xr="AvgPool",Zg="AvgPoolGrad",Jr="AvgPool3D",Qg="AvgPool3DGrad",Yr="BatchMatMul",Zr="BatchToSpaceND",Qr="Bincount",to="BitwiseAnd",eo="BroadcastArgs",Es="Cast",no="Ceil",so="ClipByValue",ro="Complex",oo="ComplexAbs",ao="Concat",io="Conv2D",co="Conv2DBackpropFilter",lo="Conv2DBackpropInput",uo="Conv3D",tm="Conv3DBackpropFilterV2",ho="Conv3DBackpropInputV2",fo="Cos",po="Cosh",go="Cumprod",mo="Cumsum",bo="CropAndResize",wo="DenseBincount",yo="DepthToSpace",$o="DepthwiseConv2dNative",ko="DepthwiseConv2dNativeBackpropFilter",xo="DepthwiseConv2dNativeBackpropInput",Eo="Diag",vo="Dilation2D",em="Dilation2DBackpropInput",nm="Dilation2DBackpropFilter",So="Draw",Io="RealDiv",To="Einsum",Do="Elu",sm="EluGrad",Ao="Erf",_o="Equal",No="Exp",Mo="ExpandDims",Co="Expm1",Bo="FFT",Fo="Fill",Ro="FlipLeftRight",Po="Floor",Oo="FloorDiv",Lo="FusedBatchNorm",Wo="GatherV2",Ko="GatherNd",Uo="Greater",Go="GreaterEqual",vs="Identity",zo="IFFT",qo="Imag",Ho="IsFinite",Vo="IsInf",jo="IsNan",Xo="LeakyRelu",Jo="Less",Yo="LessEqual",Zo="LinSpace",Qo="Log",ta="Log1p",ea="LogicalAnd",na="LogicalNot",sa="LogicalOr",ra="LRN",rm="LRNGrad",oa="Max",aa="Maximum",ia="MaxPool",om="MaxPoolGrad",ca="MaxPool3D",am="MaxPool3DGrad",la="MaxPoolWithArgmax",ua="Mean",ha="Min",fa="Minimum",da="MirrorPad",pa="Mod",ga="Multinomial",ma="Multiply",ba="Neg",wa="NotEqual",ya="NonMaxSuppressionV3",$a="NonMaxSuppressionV4",ka="NonMaxSuppressionV5",xa="OnesLike",Ea="OneHot",va="Pack",Sa="PadV2",Ia="Pow",Ta="Prelu",Da="Prod",Aa="RaggedGather",_a="RaggedRange",Na="RaggedTensorToTensor",Ma="Range",Ca="Real",Ba="Reciprocal",Fa="Relu",Ra="Reshape",Pa="ResizeNearestNeighbor",im="ResizeNearestNeighborGrad",Oa="ResizeBilinear",cm="ResizeBilinearGrad",La="Relu6",Wa="Reverse",Ka="Round",Ua="Rsqrt",Ga="ScatterNd",za="TensorScatterUpdate",qa="SearchSorted",Ha="Select",Va="Selu",ja="Slice",Xa="Sin",Ja="Sinh",Ya="Sign",Za="Sigmoid",Qa="Softplus",ti="Sqrt",ei="Sum",ni="SpaceToBatchND",si="SplitV",ri="Softmax",oi="SparseFillEmptyRows",ai="SparseReshape",ii="SparseSegmentMean",ci="SparseSegmentSum",li="SparseToDense",ui="SquaredDifference",lm="Square",hi="StaticRegexReplace",fi="StridedSlice",di="StringNGrams",pi="StringSplit",gi="StringToHashBucketFast",mi="Sub",bi="Tan",wi="Tanh",Ss="Tile",yi="TopK",$i="Transform",Ue="Transpose",ki="Unique",xi="Unpack",Ei="UnsortedSegmentSum",vi="ZerosLike",Si="Step",Hn="FromPixels",Ii="RotateWithOffset",Vn="_FusedMatMul",jn="FusedConv2D",Xn="FusedDepthwiseConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zt(...e){R().getBool("IS_TEST")||R().getBool("PROD")||console.warn(...e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ee=wn("kernelRegistry",()=>new Map),Ti=wn("gradRegistry",()=>new Map);function ve(e,t){const n=Is(e,t);return Ee.get(n)}function Jn(e){return Ti.get(e)}function Yn(e){const t=Ee.entries(),n=[];for(;;){const{done:s,value:r}=t.next();if(s)break;const[o,a]=r,[i]=o.split("_");i===e&&n.push(a)}return n}function um(e){const{kernelName:t,backendName:n}=e,s=Is(t,n);Ee.has(s)&&zt(`The kernel '${t}' for backend '${n}' is already registered`),Ee.set(s,e)}function Is(e,t){return`${t}_${e}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ts(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hm(e,t){return t==="string"?$n(e):yn([e],t)}function Di(e,t){return e instanceof Float32Array&&t==="float32"||e instanceof Int32Array&&t==="int32"||e instanceof Uint8Array&&t==="bool"}function yn(e,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=he(e)),R().getBool("DEBUG")&&Sr(e,t),Di(e,t))return e;if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool"){const n=new Uint8Array(e.length);for(let s=0;s<n.length;++s)Math.round(e[s])!==0&&(n[s]=1);return n}else throw new Error(`Unknown data type ${t}`)}function Se(){return R().platform.now()}function $n(e,t="utf-8"){return t=t||"utf-8",R().platform.encode(e,t)}function Ye(e,t="utf-8"){return t=t||"utf-8",R().platform.decode(e,t)}function lt(e){return R().platform.isTypedArray!=null?R().platform.isTypedArray(e):Ts(e)}function he(e,t=[],n=!1){if(t==null&&(t=[]),typeof e=="boolean"||typeof e=="number"||typeof e=="string"||bn(e)||e==null||lt(e)&&n)t.push(e);else if(Array.isArray(e)||lt(e))for(let s=0;s<e.length;++s)he(e[s],t,n);else{let s=-1;for(const r of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(r)&&(s=Math.max(s,Number(r)));for(let r=0;r<=s;r++)he(e[r],t,n)}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ai{constructor(t,n){this.backendTimer=t,this.logger=n,n==null&&(this.logger=new Ni)}profileKernel(t,n,s){let r;const o=()=>{r=s()};let a;const i=Se();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(o);else{o();for(const l of r)l.dataSync();a=Promise.resolve({kernelMs:Se()-i})}if(R().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let l=0;l<r.length;l++){const h=r[l];h.data().then(u=>{_i(u,h.dtype,t)})}return{kernelName:t,outputs:r,inputs:n,timeMs:a.then(l=>l.kernelMs),extraInfo:a.then(l=>l.getExtraProfileInfo!=null?l.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:n,outputs:s,timeMs:r,inputs:o,extraInfo:a}=t;s.forEach(i=>{Promise.all([i.data(),r,a]).then(c=>{this.logger.logKernelProfile(n,i,c[0],c[1],o,c[2])})})}}function _i(e,t,n){if(t!=="float32")return!1;for(let s=0;s<e.length;s++){const r=e[s];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}class Ni{logKernelProfile(t,n,s,r,o,a){const i=typeof r=="number"?we(`${r}ms`,9):r.error,c=we(t,25),l=n.rank,h=n.size,u=we(n.shape.toString(),14);let f="";for(const g in o){const w=o[g];if(w!=null){const y=w.shape||n.shape,$=y.length;f+=`${g}: ${$}D ${$>0?y:""} `}}console.log(`%c${c}	%c${i}	%c${l}D ${u}	%c${h}	%c${f}	%c${a}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mi(e,t,n){const s={},r={};for(let c=0;c<t.length;c++)s[t[c].id]=!0;for(let c=0;c<e.length;c++){const l=e[c],h=l.inputs;for(const u in h){const f=h[u];let g=!1;for(let w=0;w<t.length;w++)if(s[f.id]){l.outputs.forEach(y=>s[y.id]=!0),g=!0,r[l.id]=!0;break}if(g)break}}const o={};o[n.id]=!0;const a={};for(let c=e.length-1;c>=0;c--){const l=e[c],h=l.inputs;for(let u=0;u<l.outputs.length;u++)if(o[l.outputs[u].id]){for(const f in h)o[h[f].id]=!0,a[l.id]=!0;break}}const i=[];for(let c=0;c<e.length;c++){const l=e[c];if(r[l.id]&&a[l.id]){const h={};for(const f in l.inputs){const g=l.inputs[f];s[g.id]&&(h[f]=g)}const u=Object.assign({},l);u.inputs=h,u.outputs=l.outputs,i.push(u)}}return i}function Ci(e,t,n,s){for(let r=t.length-1;r>=0;r--){const o=t[r],a=[];if(o.outputs.forEach(c=>{const l=e[c.id];l!=null?a.push(l):a.push(null)}),o.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${o.kernelName}.`);const i=o.gradient(a);for(const c in o.inputs){if(!(c in i))throw new Error(`Cannot backprop through input ${c}. Available gradients found: ${Object.keys(i)}.`);const l=n(()=>i[c]());if(l.dtype!=="float32")throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input ${c} must have 'float32' dtype, but has '${l.dtype}'`);const h=o.inputs[c];if(!Lt(l.shape,h.shape))throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input '${c}' has shape '${l.shape}', which does not match the shape of the input '${h.shape}'`);if(e[h.id]==null)e[h.id]=l;else{const u=e[h.id];e[h.id]=s(u,l),u.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zn=20,se=3,Ge=7;function Bi(e,t,n,s){const r=ge(t),o=Fi(e,t,n,r),a=t.length,i=ye(e,t,n,r,o),c=["Tensor"];return s&&(c.push(`  dtype: ${n}`),c.push(`  rank: ${a}`),c.push(`  shape: [${t}]`),c.push("  values:")),c.push(i.map(l=>"    "+l).join(`
`)),c.join(`
`)}function Fi(e,t,n,s){const r=H(t),o=s[s.length-1],a=new Array(o).fill(0),i=t.length,c=n==="complex64"?ae(e):e;if(i>1)for(let l=0;l<r/o;l++){const h=l*o;for(let u=0;u<o;u++)a[u]=Math.max(a[u],oe(c[h+u],0,n).length)}return a}function oe(e,t,n){let s;return Array.isArray(e)?s=`${parseFloat(e[0].toFixed(Ge))} + ${parseFloat(e[1].toFixed(Ge))}j`:gn(e)?s=`'${e}'`:n==="bool"?s=Ds(e):s=parseFloat(e.toFixed(Ge)).toString(),we(s,t)}function Ds(e){return e===0?"false":"true"}function ye(e,t,n,s,r,o=!0){const a=n==="complex64"?2:1,i=t[0],c=t.length;if(c===0){if(n==="complex64"){const y=ae(e);return[oe(y[0],0,n)]}return n==="bool"?[Ds(e[0])]:[e[0].toString()]}if(c===1){if(i>Zn){const $=se*a;let E=Array.from(e.slice(0,$)),_=Array.from(e.slice((i-se)*a,i*a));return n==="complex64"&&(E=ae(E),_=ae(_)),["["+E.map((k,v)=>oe(k,r[v],n)).join(", ")+", ..., "+_.map((k,v)=>oe(k,r[i-se+v],n)).join(", ")+"]"]}return["["+(n==="complex64"?ae(e):Array.from(e)).map(($,E)=>oe($,r[E],n)).join(", ")+"]"]}const l=t.slice(1),h=s.slice(1),u=s[0]*a,f=[];if(i>Zn){for(let y=0;y<se;y++){const $=y*u,E=$+u;f.push(...ye(e.slice($,E),l,n,h,r,!1))}f.push("...");for(let y=i-se;y<i;y++){const $=y*u,E=$+u;f.push(...ye(e.slice($,E),l,n,h,r,y===i-1))}}else for(let y=0;y<i;y++){const $=y*u,E=$+u;f.push(...ye(e.slice($,E),l,n,h,r,y===i-1))}const g=c===2?",":"";f[0]="["+(i>0?f[0]+g:"");for(let y=1;y<f.length-1;y++)f[y]=" "+f[y]+g;let w=`,
`;for(let y=2;y<c;y++)w+=`
`;return f[f.length-1]=" "+f[f.length-1]+"]"+(o?"":w),f}function ae(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ze{constructor(t,n,s){if(this.dtype=n,this.shape=t.slice(),this.size=H(t),s!=null){const r=s.length;p(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||bs(n,this.size),this.strides=ge(t)}set(t,...n){n.length===0&&(n=[0]),p(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const s=this.locToIndex(n);this.values[s]=t}get(...t){t.length===0&&(t=[0]);let n=0;for(const r of t){if(r<0||r>=this.shape[n]){const o=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(o)}n++}let s=t[t.length-1];for(let r=0;r<t.length-1;++r)s+=this.strides[r]*t[r];return this.values[s]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let n=t[t.length-1];for(let s=0;s<t.length-1;++s)n+=this.strides[s]*t[s];return n}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const n=new Array(this.shape.length);for(let s=0;s<n.length-1;++s)n[s]=Math.floor(t/this.strides[s]),t-=n[s]*this.strides[s];return n[n.length-1]=t,n}get rank(){return this.shape.length}toTensor(){return ht().makeTensor(this.values,this.shape,this.dtype)}}let ht=null,qt=null;function Ri(e){ht=e}function Pi(e){qt=e}class Q{constructor(t,n,s,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=n||"float32",this.size=H(t),this.strides=ge(t),this.dataId=s,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return qt.buffer(this.shape,this.dtype,t)}bufferSync(){return qt.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return zn(this.shape,t,this.dtype==="complex64")}arraySync(){return zn(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=ht().read(this.dataId);if(this.dtype==="string"){const n=await t;try{return n.map(s=>Ye(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),ht().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=ht().readSync(this.dataId);if(this.dtype==="string")try{return t.map(n=>Ye(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await ht().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),ht().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return qt.print(this,t)}clone(){return this.throwIfDisposed(),qt.clone(this)}toString(t=!1){const n=this.dataSync();return Bi(n,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),qt.cast(this,t)}variable(t=!0,n,s){return this.throwIfDisposed(),ht().makeVariable(this,t,n,s)}}Object.defineProperty(Q,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function As(){return wn("Tensor",()=>Q)}As();class Ie extends Q{constructor(t,n,s,r){super(t.shape,t.dtype,t.dataId,r),this.trainable=n,this.name=s}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!Lt(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);ht().disposeTensor(this),this.dataId=t.dataId,ht().incRef(this,null)}dispose(){ht().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(Ie,Symbol.hasInstance,{value:e=>e instanceof Q&&e.assign!=null&&e.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Qn;(function(e){e.R0="R0",e.R1="R1",e.R2="R2",e.R3="R3",e.R4="R4",e.R5="R5",e.R6="R6"})(Qn||(Qn={}));var Qe;(function(e){e.float32="float32",e.int32="int32",e.bool="int32",e.complex64="complex64"})(Qe||(Qe={}));var tn;(function(e){e.float32="float32",e.int32="int32",e.bool="bool",e.complex64="complex64"})(tn||(tn={}));var en;(function(e){e.float32="float32",e.int32="float32",e.bool="float32",e.complex64="complex64"})(en||(en={}));var nn;(function(e){e.float32="complex64",e.int32="complex64",e.bool="complex64",e.complex64="complex64"})(nn||(nn={}));const Oi={float32:en,int32:Qe,bool:tn,complex64:nn};function _s(e,t){if(e==="string"||t==="string"){if(e==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return Oi[e][t]}function fm(e){return _s(e,"int32")}function Ns(e){return e!=null&&typeof e=="object"&&"texture"in e&&e.texture instanceof WebGLTexture}function Ms(e){return typeof GPUBuffer<"u"&&e!=null&&typeof e=="object"&&"buffer"in e&&e.buffer instanceof GPUBuffer}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function V(e,t){if(e.dtype===t.dtype)return[e,t];const n=_s(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function Li(e,t){p(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function Cs(e){const t=[];return Bs(e,t,new Set),t}function Bs(e,t,n){if(e==null)return;if(e instanceof Q){t.push(e);return}if(!Wi(e))return;const s=e;for(const r in s){const o=s[r];n.has(o)||(n.add(o),Bs(o,t,n))}}function Wi(e){return Array.isArray(e)||typeof e=="object"}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ze(e){return e.kernelName!=null}class ts{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class Yt{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new ts}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:n}=this.initializeBackend(t);if(n)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,n,s=1){return t in this.registryFactory?(zt(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:n,priority:s},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:n,asyncInit:s}=this.initializeBackend(t);if(!(s?await n:n))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new Ai(this.backendInstance),!0}setupRegisteredKernels(){Yn(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){Yn(t).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[t])})}initializeBackend(t){const n=this.registryFactory[t];if(n==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const s=n.factory();if(s&&!(s instanceof kr)&&typeof s.then=="function"){const r=++this.pendingBackendInitId,o=s.then(a=>r<this.pendingBackendInitId?!1:(this.registry[t]=a,this.pendingBackendInit=null,!0)).catch(a=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,zt(`Initialization of backend ${t} failed`),zt(a.stack||a.message)),!1));return this.pendingBackendInit=o,{success:o,asyncInit:!0}}else return this.registry[t]=s,{success:!0,asyncInit:!1}}catch(s){return zt(`Initialization of backend ${t} failed`),zt(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,n)=>this.registryFactory[n].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n],{success:r,asyncInit:o}=this.initializeBackend(s);if(o||r)return{name:s,asyncInit:o}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,n){const s=this.state.tensorInfo.get(n),r=s.backend,o=this.readSync(n),a=r.refCount(n);r.disposeData(n,!0),s.backend=t,t.move(n,o,s.shape,s.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,n){let s=null;if(n==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");n=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=t}let r;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(r),()=>(r=n(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(t,n,s){t();try{const r=s();return n(),r}catch(r){throw n(),r}}nextTensorId(){return Yt.nextTensorId++}nextVariableId(){return Yt.nextVariableId++}clone(t){const n=b.runKernel(vs,{x:t}),s={x:t},r=a=>({x:()=>{const i="float32",c={x:a},l={dtype:i};return b.runKernel(Es,c,l)}}),o=[];return this.addTapeNode(this.state.activeScope.name,s,[n],r,o,{}),n}runKernel(t,n,s){if(this.backendName==null&&this.backend,!(ve(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:n,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,n,s){const r=this.backend.numDataIds();let o=0;s.forEach(c=>{o+=c.dtype==="complex64"?3:1});const a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],i=r-n-o-a;if(i>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${i} data ids) after running '${t}'`)}runKernelFunc(t){let n,s=[];const r=this.isTapeOn(),o=this.state.numBytes,a=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let i;this.backendName==null&&this.backend;let c;const l=ze(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(ze(t)){const{kernelName:w,inputs:y,attrs:$}=t;this.backendName==null&&this.backend;const E=ve(w,this.backendName);p(E!=null,()=>`Cannot find registered kernel '${w}' for backend '${this.backendName}'`),i=()=>{const _=this.backend.numDataIds();c=E.kernelFunc({inputs:y,attrs:$,backend:this.backend});const k=Array.isArray(c)?c:[c];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(w,_,k);const v=k.map(S=>S.rank!=null?S:this.makeTensorFromTensorInfo(S));if(r){const S=this.getTensorsForGradient(w,y,v);s=this.saveTensorsForBackwardMode(S)}return v}}else{const{forwardFunc:w}=t,y=$=>{r&&(s=$.map(E=>this.keep(this.clone(E))))};i=()=>{const $=this.backend.numDataIds();c=this.tidy(()=>w(this.backend,y));const E=Array.isArray(c)?c:[c];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(l,$,E),E}}const{inputs:h,attrs:u}=t,f=ze(t)?null:t.backwardsFunc;let g;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=i():(g=this.profiler.profileKernel(l,h,()=>i()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(g),n=g.outputs)}),r&&this.addTapeNode(l,h,n,f,s,u),this.state.profiling&&this.state.activeProfile.kernels.push({name:l,bytesAdded:this.state.numBytes-o,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(h).map(w=>h[w]!=null?h[w].shape:null),outputShapes:n.map(w=>w.shape),kernelTimeMs:g.timeMs,extraInfo:g.extraInfo}),Array.isArray(c)?n:n[0]}saveTensorsForBackwardMode(t){return t.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(t,n,s){const r=Jn(t);if(r!=null){const o=r.inputsToSave||[],a=r.outputsToSave||[];let i;r.saveAllInputs?(p(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),i=Object.keys(n).map(l=>n[l])):i=o.map(l=>n[l]);const c=s.filter((l,h)=>a[h]);return i.concat(c)}return[]}makeTensor(t,n,s,r){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",r=r||this.backend;let o=t;s==="string"&&gn(t[0])&&(o=t.map(c=>$n(c)));const a=r.write(o,n,s),i=new Q(n,s,a,this.nextTensorId());if(this.trackTensor(i,r),s==="string"){const c=this.state.tensorInfo.get(a),l=Tr(o);this.state.numBytes+=l-c.bytes,c.bytes=l}return i}makeTensorFromDataId(t,n,s,r){s=s||"float32";const o={dataId:t,shape:n,dtype:s};return this.makeTensorFromTensorInfo(o,r)}makeTensorFromTensorInfo(t,n){const{dataId:s,shape:r,dtype:o}=t,a=new Q(r,o,s,this.nextTensorId());return this.trackTensor(a,n),a}makeVariable(t,n=!0,s,r){s=s||this.nextVariableId().toString(),r!=null&&r!==t.dtype&&(t=t.cast(r));const o=new Ie(t,n,s,this.nextTensorId());if(this.state.registeredVariables[o.name]!=null)throw new Error(`Variable with name ${o.name} was already registered`);return this.state.registeredVariables[o.name]=o,this.incRef(o,this.backend),o}trackTensor(t,n){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let s=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(s=t.size*Xe(t.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:n||this.backend,dtype:t.dtype,shape:t.shape,bytes:s})),t instanceof Ie||this.track(t)}incRef(t,n){this.trackTensor(t,n),this.backend.incRef(t.dataId)}removeDataId(t,n){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===n&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const n=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const s=t.size*Xe(t.dtype);this.state.numBytes-=s}n.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,n.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const n=this.state.registeredVariables[t];this.disposeVariable(n)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const n=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,n,s,r,o,a){const i={id:this.state.nextTapeNodeId++,kernelName:t,inputs:n,outputs:s,saved:o},c=Jn(t);c!=null&&(r=c.gradFunc),r!=null&&(i.gradient=l=>(l=l.map((h,u)=>{if(h==null){const f=s[u],g=mn(f.size,f.dtype);return this.makeTensor(g,f.shape,f.dtype)}return h}),r(l.length>1?l:l[0],o,a))),this.state.activeTape.push(i)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(n.name=t),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(t){const n=Cs(t),s=new Set(n.map(o=>o.id));for(let o=0;o<this.state.activeScope.track.length;o++){const a=this.state.activeScope.track[o];!a.kept&&!s.has(a.id)&&a.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(o=>{!o.kept&&o.scopeId===r.id&&this.track(o)})}gradients(t,n,s,r=!1){if(p(n.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const o=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));p(o instanceof Q,()=>"The result y returned by f() must be a tensor.");const a=Mi(this.state.activeTape,n,o);if(!r&&a.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const i={};i[o.id]=s??Ki(o.shape),Ci(i,a,l=>this.tidy(l),Ui);const c=n.map(l=>i[l.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(l=>{for(const h of l.saved)h.dispose()}),this.state.activeTape=null),{value:o,grads:c}})}customGrad(t){return p(Je(t),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{p(n.every(i=>i instanceof Q),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const r={};n.forEach((i,c)=>{r[c]=i});const o=(i,c)=>(s=t(...n,c),p(s.value instanceof Q,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),p(Je(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),a=(i,c)=>{const l=s.gradFunc(i,c),h=Array.isArray(l)?l:[l];p(h.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),p(h.every(f=>f instanceof Q),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const u={};return h.forEach((f,g)=>{u[g]=()=>f}),u};return this.runKernelFunc({forwardFunc:o,backwardsFunc:a,inputs:r})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,n){return this.state.tensorInfo.get(t).backend.readToGPU(t,n)}async time(t){const n=Se(),s=await this.backend.time(t);return s.wallMs=Se()-n,s}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new ts;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}Yt.nextTensorId=0;Yt.nextVariableId=0;function Ki(e){const t=ys(H(e),"float32");return b.makeTensor(t,e,"float32")}function Fs(){const e=ks();if(e._tfengine==null){const t=new _r(e);e._tfengine=new Yt(t)}return Br(e._tfengine.ENV),Ri(()=>e._tfengine),e._tfengine}const b=Fs();function Ui(e,t){const n={a:e,b:t};return b.runKernel(xs,n)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gi(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const et=R();et.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});et.registerFlag("IS_BROWSER",()=>Gi());et.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");et.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));et.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));et.registerFlag("PROD",()=>!1);et.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>et.getBool("DEBUG"));et.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);et.registerFlag("IS_TEST",()=>!1);et.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>et.getBool("DEBUG"));et.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);et.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);et.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function St(e,t){let n=e;if(lt(e))return t==="string"?[]:[e.length];if(Ns(e)){const r=e.channels||"RGBA";return[e.height,e.width*r.length]}else if(Ms(e))return[e.buffer.size/(t==null?4:Xe(t))];if(!Array.isArray(e))return[];const s=[];for(;Array.isArray(n)||lt(n)&&t!=="string";)s.push(n.length),n=n[0];return Array.isArray(e)&&R().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&Rs(e,s,[]),s}function Rs(e,t,n){if(n=n||[],!Array.isArray(e)&&!lt(e)){p(t.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}p(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),p(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const s=t.slice(1);for(let r=0;r<e.length;++r)Rs(e[r],s,n.concat(r))}function es(e,t,n,s){if(e!=="string_or_numeric"){if(e==null)throw new Error("Expected dtype cannot be null.");if(e!=="numeric"&&e!==t||e==="numeric"&&t==="string")throw new Error(`Argument '${n}' passed to '${s}' must be ${e} tensor, but got ${t} tensor`)}}function d(e,t,n,s="numeric"){if(e instanceof As())return es(s,e.dtype,t,n),e;let r=Re(e);if(r!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(r=s),es(s,r,t,n),e==null||!lt(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string"){const c=e==null?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${c}'`)}const o=St(e,r);!lt(e)&&!Array.isArray(e)&&(e=[e]);const i=r!=="string"?yn(e,r):he(e,[],!0);return b.makeTensor(i,o,r)}function Te(e,t,n,s="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((o,a)=>d(o,`${t}[${a}]`,n,s))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zi="__op";function m(e){const t=Object.keys(e);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const s=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+zi;const r=(...o)=>{b.startScope(n);try{const a=s(...o);return bn(a)&&console.error("Cannot return a Promise inside of tidy."),b.endScope(a),a}catch(a){throw b.endScope(null),a}};return Object.defineProperty(r,"name",{value:n,configurable:!0}),r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qi(e,t){const n=d(e,"real","complex"),s=d(t,"imag","complex");ut(n.shape,s.shape,`real and imag shapes, ${n.shape} and ${s.shape}, must match in call to tf.complex().`);const r={real:n,imag:s};return b.runKernel(ro,r)}const Ct=m({complex_:qi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function It(e,t,n,s){if(s==null)s=Re(e);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(Ms(e)||Ns(e)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return b.backend.createTensorFromGPUData(e,t||n,s)}if(!lt(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){ot(t);const r=H(t),o=H(n);p(r===o,()=>`Based on the provided shape, [${t}], the tensor should have ${r} values but has ${o}`);for(let a=0;a<n.length;++a){const i=n[a],c=a===n.length-1?i!==H(t.slice(a)):!0;p(n[a]===t[a]||!c,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!lt(e)&&!Array.isArray(e)&&(e=[e]),t=t||n,e=s!=="string"?yn(e,s):he(e,[],!0),b.makeTensor(e,t,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ie(e,t,n){const s=St(e,n);return It(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bt={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class dt{static join(t){return new dt(t).slice()}constructor(t){if(this.shards=[],this.previousShardIndex=0,t==null||(t instanceof Array||(t=[t]),t=t.map(s=>lt(s)?s.buffer:s),t.length===0))return;this.bufferUniformSize=t[0].byteLength;let n=0;for(let s=0;s<t.length;s++){const r=t[s];s!==t.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const o=n+r.byteLength;this.shards.push({buffer:r,start:n,end:o}),n=o}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(t=0,n=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(t=isNaN(Number(t))?0:t,n=isNaN(Number(n))?0:n,t=Math.max(0,t),n=Math.min(this.byteLength,n),n<=t)return new ArrayBuffer(0);const s=this.findShardForByte(t);if(s===-1)throw new Error(`Could not find start shard for byte ${t}`);const r=n-t,o=new ArrayBuffer(r),a=new Uint8Array(o);let i=0;for(let c=s;c<this.shards.length;c++){const l=this.shards[c],u=t+i-l.start,f=i,w=Math.min(n,l.end)-l.start,y=new Uint8Array(l.buffer,u,w-u);if(a.set(y,f),i+=y.length,n<l.end)break}return o}findShardForByte(t){if(this.shards.length===0||t<0||t>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(t/this.bufferUniformSize),this.previousShardIndex;function n(r){return t<r.start?-1:t>=r.end?1:0}if(n(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=Hi(this.shards,n);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function Hi(e,t){let n=0,s=e.length;for(;n<=s;){const r=Math.floor((s-n)/2)+n,o=t(e[r]);if(o===0)return r;o<0?s=r:n=r+1}return-1}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dm(){return b}function X(e,t){return b.tidy(e,t)}function st(e){Cs(e).forEach(n=>n.dispose())}function Vi(e){return b.keep(e)}function pm(){return b.ready()}function ji(){return b.backendName}function gm(e,t,n=1){return b.registerBackend(e,t,n)}function Xi(){return b.backend}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xt=4;async function Ji(e,t){const n=[],s=[],r=Array.isArray(e)?e.map(a=>a.name):Object.keys(e);for(let a=0;a<r.length;++a){const i=r[a],c=Array.isArray(e)?e[a].tensor:e[i];if(c.dtype!=="float32"&&c.dtype!=="int32"&&c.dtype!=="bool"&&c.dtype!=="string"&&c.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${i}': ${c.dtype}`);const l={name:i,shape:c.shape,dtype:c.dtype};if(c.dtype==="string"){const h=new Promise(async u=>{const f=await c.bytes(),g=f.reduce(($,E)=>$+E.length,0)+xt*f.length,w=new Uint8Array(g);let y=0;for(let $=0;$<f.length;$++){const E=f[$],_=new Uint8Array(new Uint32Array([E.length]).buffer);w.set(_,y),y+=xt,w.set(E,y),y+=E.length}u(w)});s.push(h)}else s.push(c.data());t!=null&&(l.group=t),n.push(l)}const o=await Promise.all(s);return{data:tc(o),specs:n}}function Ps(e,t){const n=new dt(e),s={};let r=0;for(const o of t){const a=Yi(o,(i,c)=>n.slice(r+i,r+c));s[o.name]=Os(o,n.slice(r,r+a)),r+=a}return s}function Yi(e,t){const n=H(e.shape);let s;if("quantization"in e){const r=e.quantization;s=Bt[r.dtype]}else if(e.dtype==="string"){let r=0;for(let o=0;o<n;o++)r+=xt+new Uint32Array(t(r,r+xt))[0];return r}else s=Bt[e.dtype];return n*s}async function Zi(e,t){const n=H(e.shape);let s;if("quantization"in e){const r=e.quantization;s=Bt[r.dtype]}else if(e.dtype==="string"){let r=0;for(let o=0;o<n;o++)r+=xt+new Uint32Array(await t(r,r+xt))[0];return r}else s=Bt[e.dtype];return n*s}function Os(e,t){const n=e.name,s=e.dtype,r=e.shape,o=H(r);let a,i=0;if("quantization"in e){const c=e.quantization;if(c.dtype==="uint8"||c.dtype==="uint16"){if(!("min"in c&&"scale"in c))throw new Error(`Weight ${e.name} with quantization ${c.dtype} doesn't have corresponding metadata min and scale.`)}else if(c.dtype==="float16"){if(s!=="float32")throw new Error(`Weight ${e.name} is quantized with ${c.dtype} which only supports weights of type float32 not ${s}.`)}else throw new Error(`Weight ${e.name} has unknown quantization dtype ${c.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const l=Bt[c.dtype],h=c.dtype==="uint8"?new Uint8Array(t):new Uint16Array(t);if(s==="float32")if(c.dtype==="uint8"||c.dtype==="uint16"){a=new Float32Array(h.length);for(let u=0;u<h.length;u++){const f=h[u];a[u]=f*c.scale+c.min}}else if(c.dtype==="float16")a=ic()(h);else throw new Error(`Unsupported quantization type ${c.dtype} for weight type float32.`);else if(s==="int32"){if(c.dtype!=="uint8"&&c.dtype!=="uint16")throw new Error(`Unsupported quantization type ${c.dtype} for weight type int32.`);a=new Int32Array(h.length);for(let u=0;u<h.length;u++){const f=h[u];a[u]=Math.round(f*c.scale+c.min)}}else throw new Error(`Unsupported dtype in weight '${n}': ${s}`);i+=o*l}else if(s==="string"){const c=H(e.shape);a=[];for(let l=0;l<c;l++){const h=new Uint32Array(t.slice(i,i+xt))[0];i+=xt;const u=new Uint8Array(t.slice(i,i+h));a.push(u),i+=h}}else{const c=Bt[s];if(s==="float32")a=new Float32Array(t);else if(s==="int32")a=new Int32Array(t);else if(s==="bool")a=new Uint8Array(t);else if(s==="complex64"){a=new Float32Array(t);const l=new Float32Array(a.length/2),h=new Float32Array(a.length/2);for(let w=0;w<l.length;w++)l[w]=a[w*2],h[w]=a[w*2+1];const u=ie(l,r,"float32"),f=ie(h,r,"float32"),g=Ct(u,f);return u.dispose(),f.dispose(),g}else throw new Error(`Unsupported dtype in weight '${n}': ${s}`);i+=o*c}return ie(a,r,s)}async function ns(e,t,n){let s=new Uint8Array(t);for(;s.byteLength<n;){const{done:r,value:o}=await e.read();if(r&&o==null){const i=n-s.byteLength;throw new Error(`Reader is done but ${i} bytes are still expected`)}const a=new Uint8Array(s.length+o.byteLength);a.set(s,0),a.set(new Uint8Array(o),s.length),s=a}return s.buffer}async function Qi(e,t){const n={},s=e.getReader();let r=new ArrayBuffer(0);for(const o of t){const a=await Zi(o,async(l,h)=>(r=await ns(s,r,h),r.slice(l,h)));r=await ns(s,r,a);const i=r.slice(0,a);r=r.slice(a);const c=Os(o,i);if(n[o.name]=c,ji()==="webgpu"){const l=Xi();"uploadToGPU"in l&&H(c.shape)>=R().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&l.uploadToGPU(c.dataId)}}return n}function tc(e){if(e===null)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(o=>{if(t+=o.byteLength,n.push(o.byteLength===o.buffer.byteLength?o:new o.constructor(o)),!(o instanceof Float32Array||o instanceof Int32Array||o instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${o.constructor.name}`)});const s=new Uint8Array(t);let r=0;return n.forEach(o=>{s.set(new Uint8Array(o.buffer),r),r+=o.byteLength}),s.buffer}const kn=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function ss(e){return kn?Buffer.byteLength(e,"utf8"):new Blob([e]).size}function ec(e){if(kn)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let s=0,r=t.length;s<r;s++)n+=String.fromCharCode(t[s]);return btoa(n)}function nc(e){if(kn){const s=Buffer.from(e,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let s=0;s<t.length;++s)n.set([t.charCodeAt(s)],s);return n.buffer}function sc(e){return dt.join(e)}function rs(e){const t="/";for(e=e.trim();e.endsWith(t);)e=e.slice(0,e.length-1);const n=e.split(t);return n[n.length-1]}function Ls(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function Ws(e,t,n){const s={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(s.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=t,s.weightData=n}return e.signature!=null&&(s.signature=e.signature),e.userDefinedMetadata!=null&&(s.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(s.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(s.initializerSignature=e.initializerSignature),s}async function xn(e,t){let n,s;return e.weightsManifest!=null&&([n,s]=await t(e.weightsManifest)),Ws(e,n,s)}function me(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:e.modelTopology==null?0:ss(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:ss(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new dt(e.weightData).byteLength}}function sn(e){const t=[];for(const n of e)t.push(...n.weights);return t}function rc(){const e=n=>{let s=n<<13,r=0;for(;!(s&8388608);)r-=8388608,s<<=1;return s&=-8388609,r+=947912704,s|r},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}function oc(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function ac(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function ic(){const e=rc(),t=oc(),n=ac();return s=>{const r=new ArrayBuffer(4*s.length),o=new Uint32Array(r);for(let a=0;a<s.length;a++){const i=s[a],c=e[n[i>>10]+(i&1023)]+t[i>>10];o[a]=c}return new Float32Array(r)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class z{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return z.instance==null&&(z.instance=new z),z.instance}static registerSaveRouter(t){z.getInstance().saveRouters.push(t)}static registerLoadRouter(t){z.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return z.getHandlers(t,"save")}static getLoadHandlers(t,n){return z.getHandlers(t,"load",n)}static getHandlers(t,n,s){const r=[];return(n==="load"?z.getInstance().loadRouters:z.getInstance().saveRouters).forEach(a=>{const i=a(t,s);i!==null&&r.push(i)}),r}}const cc=e=>z.registerSaveRouter(e),lc=e=>z.registerLoadRouter(e),uc=e=>z.getSaveHandlers(e),hc=(e,t)=>z.getLoadHandlers(e,t);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rn="tensorflowjs",on=1,Nt="models_store",kt="model_info_store";function Ks(){if(!R().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e=typeof window>"u"?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function an(e){const t=e.result;t.createObjectStore(Nt,{keyPath:"modelPath"}),t.createObjectStore(kt,{keyPath:"modelPath"})}class Ft{constructor(t){if(this.indexedDB=Ks(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,n){return new Promise((s,r)=>{const o=this.indexedDB.open(rn,on);o.onupgradeneeded=()=>an(o),o.onsuccess=()=>{const a=o.result;if(n==null){const i=a.transaction(Nt,"readonly"),l=i.objectStore(Nt).get(this.modelPath);l.onsuccess=()=>{if(l.result==null)return a.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(l.result.modelArtifacts)},l.onerror=h=>(a.close(),r(l.error)),i.oncomplete=()=>a.close()}else{n.weightData=dt.join(n.weightData);const i=me(n),c=a.transaction(kt,"readwrite");let l=c.objectStore(kt),h;try{h=l.put({modelPath:this.modelPath,modelArtifactsInfo:i})}catch(f){return r(f)}let u;h.onsuccess=()=>{u=a.transaction(Nt,"readwrite");const f=u.objectStore(Nt);let g;try{g=f.put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:i})}catch(w){return r(w)}g.onsuccess=()=>s({modelArtifactsInfo:i}),g.onerror=w=>{l=c.objectStore(kt);const y=l.delete(this.modelPath);y.onsuccess=()=>(a.close(),r(g.error)),y.onerror=$=>(a.close(),r(g.error))}},h.onerror=f=>(a.close(),r(h.error)),c.oncomplete=()=>{u==null?a.close():u.oncomplete=()=>a.close()}}},o.onerror=a=>r(o.error)})}}Ft.URL_SCHEME="indexeddb://";const Us=e=>R().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Ft.URL_SCHEME)?fc(e.slice(Ft.URL_SCHEME.length)):null;z.registerSaveRouter(Us);z.registerLoadRouter(Us);function fc(e){return new Ft(e)}function dc(e){return e.startsWith(Ft.URL_SCHEME)?e.slice(Ft.URL_SCHEME.length):e}class pc{constructor(){this.indexedDB=Ks()}async listModels(){return new Promise((t,n)=>{const s=this.indexedDB.open(rn,on);s.onupgradeneeded=()=>an(s),s.onsuccess=()=>{const r=s.result,o=r.transaction(kt,"readonly"),i=o.objectStore(kt).getAll();i.onsuccess=()=>{const c={};for(const l of i.result)c[l.modelPath]=l.modelArtifactsInfo;t(c)},i.onerror=c=>(r.close(),n(i.error)),o.oncomplete=()=>r.close()},s.onerror=r=>n(s.error)})}async removeModel(t){return t=dc(t),new Promise((n,s)=>{const r=this.indexedDB.open(rn,on);r.onupgradeneeded=()=>an(r),r.onsuccess=()=>{const o=r.result,a=o.transaction(kt,"readwrite"),i=a.objectStore(kt),c=i.get(t);let l;c.onsuccess=()=>{if(c.result==null)return o.close(),s(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const h=i.delete(t),u=()=>{l=o.transaction(Nt,"readwrite");const g=l.objectStore(Nt).delete(t);g.onsuccess=()=>n(c.result.modelArtifactsInfo),g.onerror=w=>s(c.error)};h.onsuccess=u,h.onerror=f=>(u(),o.close(),s(c.error))}},c.onerror=h=>(o.close(),s(c.error)),a.oncomplete=()=>{l==null?o.close():l.oncomplete=()=>o.close()}},r.onerror=o=>s(r.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bt="/",Ht="tensorflowjs_models",Gs="info",gc="model_topology",mc="weight_specs",bc="weight_data",wc="model_metadata";function zs(e){return{info:[Ht,e,Gs].join(bt),topology:[Ht,e,gc].join(bt),weightSpecs:[Ht,e,mc].join(bt),weightData:[Ht,e,bc].join(bt),modelMetadata:[Ht,e,wc].join(bt)}}function qs(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function yc(e){const t=e.split(bt);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(bt)}function $c(e){return e.startsWith(Rt.URL_SCHEME)?e.slice(Rt.URL_SCHEME.length):e}class Rt{constructor(t){if(!R().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=zs(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(t.modelTopology),s=JSON.stringify(t.weightSpecs),r=me(t),o=dt.join(t.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,ec(o));const a={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:r}}catch{throw qs(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=s;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=r;const o=this.LS.getItem(this.keys.modelMetadata);if(o!=null){const i=JSON.parse(o);n.format=i.format,n.generatedBy=i.generatedBy,n.convertedBy=i.convertedBy,i.signature!=null&&(n.signature=i.signature),i.userDefinedMetadata!=null&&(n.userDefinedMetadata=i.userDefinedMetadata),i.modelInitializer!=null&&(n.modelInitializer=i.modelInitializer),i.initializerSignature!=null&&(n.initializerSignature=i.initializerSignature),i.trainingConfig!=null&&(n.trainingConfig=i.trainingConfig)}const a=this.LS.getItem(this.keys.weightData);if(a==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=nc(a),n}}Rt.URL_SCHEME="localstorage://";const Hs=e=>R().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Rt.URL_SCHEME)?kc(e.slice(Rt.URL_SCHEME.length)):null;z.registerSaveRouter(Hs);z.registerLoadRouter(Hs);function kc(e){return new Rt(e)}class xc{constructor(){p(R().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),p(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},n=Ht+bt,s=bt+Gs;for(let r=0;r<this.LS.length;++r){const o=this.LS.key(r);if(o.startsWith(n)&&o.endsWith(s)){const a=yc(o);t[a]=JSON.parse(this.LS.getItem(o))}}return t}async removeModel(t){t=$c(t);const n=zs(t);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${t}'`);const s=JSON.parse(this.LS.getItem(n.info));return qs(n),s}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jt="://";class Z{constructor(){this.managers={}}static getInstance(){return Z.instance==null&&(Z.instance=new Z),Z.instance}static registerManager(t,n){p(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(jt)&&(t=t.slice(0,t.indexOf(jt))),p(t.length>0,()=>"scheme must not be an empty string.");const s=Z.getInstance();p(s.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),s.managers[t]=n}static getManager(t){const n=Z.getInstance().managers[t];if(n==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(Z.getInstance().managers)}}function $e(e){if(e.indexOf(jt)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${Z.getSchemes().join(",")}`);return{scheme:e.split(jt)[0],path:e.split(jt)[1]}}async function Vs(e,t,n=!1){p(e!==t,()=>`Old path and new path are the same: '${e}'`);const s=z.getLoadHandlers(e);p(s.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),p(s.length<2,()=>`Copying failed because more than one (${s.length}) load handlers for source URL ${e}.`);const r=s[0],o=z.getSaveHandlers(t);p(o.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),p(o.length<2,()=>`Copying failed because more than one (${s.length}) save handlers for destination URL ${t}.`);const a=o[0],i=$e(e).scheme,c=$e(e).path,l=i===$e(e).scheme,h=await r.load();n&&l&&await Z.getManager(i).removeModel(c);const u=await a.save(h);return n&&!l&&await Z.getManager(i).removeModel(c),u.modelArtifactsInfo}async function Ec(){const e=Z.getSchemes(),t={};for(const n of e){const s=await Z.getManager(n).listModels();for(const r in s){const o=n+jt+r;t[o]=s[r]}}return t}async function vc(e){const t=$e(e);return Z.getManager(t.scheme).removeModel(t.path)}async function Sc(e,t){return Vs(e,t,!1)}async function Ic(e,t){return Vs(e,t,!0)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Tc{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,n){return fetch(t,n)}now(){return performance.now()}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,n){return new TextDecoder(n).decode(t)}setTimeoutCustom(t,n){if(typeof window>"u"||!R().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,n);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const r=this.functionRefs[s.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return Ts(t)}}if(R().get("IS_BROWSER")){R().setPlatform("browser",new Tc);try{Z.registerManager(Rt.URL_SCHEME,new xc)}catch{}try{Z.registerManager(Ft.URL_SCHEME,new pc)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dc={importFetch:()=>require("node-fetch")};let qe;class Ac{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return R().global.fetch!=null?R().global.fetch(t,n):(qe==null&&(qe=Dc.importFetch()),qe(t,n))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(t)}decode(t,n){return t.length===0?"":new this.util.TextDecoder(n).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}R().get("IS_NODE")&&!R().get("IS_BROWSER")&&R().setPlatform("node",new Ac);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Et(e,t="float32",n){return t=t||"float32",ot(e),new Ze(e,t,n)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _c(e,t){const n=d(e,"x","cast");if(!Ir(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&n.dtype!=="string"||t!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:n},r={dtype:t};return b.runKernel(Es,s,r)}const j=m({cast_:_c});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nc(e){const n={x:d(e,"x","clone","string_or_numeric")};return b.runKernel(vs,n)}const Xt=m({clone_:Nc});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mc(e,t=!1){console.log(e.toString(t))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Fs();const Cc={buffer:Et,cast:j,clone:Xt,print:Mc};Pi(Cc);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bc(e,t){let n=d(e,"a","add"),s=d(t,"b","add");[n,s]=V(n,s);const r={a:n,b:s};return b.runKernel(xs,r)}const M=m({add_:Bc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fc(e,t){let n=d(e,"a","floorDiv"),s=d(t,"b","floorDiv");[n,s]=V(n,s);const r={a:n,b:s};return b.runKernel(Oo,r)}const Rc=m({floorDiv_:Fc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pc(e,t){let n=d(e,"a","div"),s=d(t,"b","div");if([n,s]=V(n,s),n.dtype==="int32"&&s.dtype==="int32")return Rc(n,s);const r={a:n,b:s},o={};return b.runKernel(Io,r,o)}const G=m({div_:Pc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oc(e,t){let n=d(e,"a","mul"),s=d(t,"b","mul");[n,s]=V(n,s);const r={a:n,b:s};return b.runKernel(ma,r)}const I=m({mul_:Oc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lc(e){const t=d(e,"x","abs");if(t.dtype==="complex64"){const n={x:t};return b.runKernel(oo,n)}else{const n={x:t};return b.runKernel(Rr,n)}}const it=m({abs_:Lc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wc(e){const n={x:d(e,"x","acos")};return b.runKernel(Pr,n)}const mm=m({acos_:Wc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kc(e){const n={x:d(e,"x","acosh")};return b.runKernel(Or,n)}const bm=m({acosh_:Kc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uc(e){p(Array.isArray(e),()=>"The argument passed to tf.addN() must be a list of tensors"),p(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);const t=e.map((r,o)=>d(r,`tensors${o}`,"addN")),n=t[0];t.forEach(r=>{if(r.dtype!==n.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),t.forEach(r=>{if(!Lt(r.shape,n.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const s=t;return b.runKernel(Lr,s)}const wm=m({addN_:Uc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gc(e,t=null,n=!1){const r={x:d(e,"x","all","bool")},o={axis:t,keepDims:n};return b.runKernel(Wr,r,o)}const ym=m({all_:Gc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zc(e,t=null,n=!1){const r={x:d(e,"x","any","bool")},o={axis:t,keepDims:n};return b.runKernel(Kr,r,o)}const $m=m({any_:zc});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qc(e,t=0){const s={x:d(e,"x","argMax")},r={axis:t};return b.runKernel(Ur,s,r)}const km=m({argMax_:qc});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hc(e,t=0){const s={x:d(e,"x","argMin")},r={axis:t};return b.runKernel(Gr,s,r)}const xm=m({argMin_:Hc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vc(e){const n={x:d(e,"x","asin")};return b.runKernel(zr,n)}const Em=m({asin_:Vc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jc(e){const n={x:d(e,"x","asinh")};return b.runKernel(qr,n)}const vm=m({asinh_:jc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xc(e){const n={x:d(e,"x","atan")};return b.runKernel(Hr,n)}const Sm=m({atan_:Xc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jc(e,t){let n=d(e,"a","atan2"),s=d(t,"b","atan2");[n,s]=V(n,s);const r={a:n,b:s};return b.runKernel(jr,r)}const Im=m({atan2_:Jc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yc(e){const n={x:d(e,"x","atanh")};return b.runKernel(Vr,n)}const Tm=m({atanh_:Yc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dm(e,t,n,s,r="NHWC",o){const a=e[3],i=[...t,a],c=rl(r);return Pe(e,i,n,o,s,null,null,c)}function Zc(e,t,n,s,r,o,a="channelsLast"){const[i,c]=fe(t);let l;if(a==="channelsLast")l=[i,c,e[3],e[3]];else if(a==="channelsFirst")l=[i,c,e[1],e[1]];else throw new Error(`Unknown dataFormat ${a}`);return Pe(e,l,n,s,r,o,!1,a)}function Am(e,t,n,s,r,o,a="NDHWC"){const[i,c,l]=cn(t);let h,u;if(a==="NDHWC")u="channelsLast",h=[i,c,l,e[4],e[4]];else if(a==="NCDHW")u="channelsFirst",h=[i,c,l,e[1],e[1]];else throw new Error(`Unknown dataFormat ${a}`);return Qc(e,h,n,s,r,!1,u,o)}function Pe(e,t,n,s,r,o,a=!1,i="channelsLast"){let[c,l,h,u]=[-1,-1,-1,-1];if(i==="channelsLast")[c,l,h,u]=e;else if(i==="channelsFirst")[c,u,l,h]=e;else throw new Error(`Unknown dataFormat ${i}`);const[f,g,,w]=t,[y,$]=fe(n),[E,_]=fe(s),k=Jt(f,E),v=Jt(g,_),{padInfo:S,outHeight:T,outWidth:N}=nl(r,l,h,y,$,k,v,o,i),D=a?w*u:w;let A;return i==="channelsFirst"?A=[c,D,T,N]:i==="channelsLast"&&(A=[c,T,N,D]),{batchSize:c,dataFormat:i,inHeight:l,inWidth:h,inChannels:u,outHeight:T,outWidth:N,outChannels:D,padInfo:S,strideHeight:y,strideWidth:$,filterHeight:f,filterWidth:g,effectiveFilterHeight:k,effectiveFilterWidth:v,dilationHeight:E,dilationWidth:_,inShape:e,outShape:A,filterShape:t}}function Qc(e,t,n,s,r,o=!1,a="channelsLast",i){let[c,l,h,u,f]=[-1,-1,-1,-1,-1];if(a==="channelsLast")[c,l,h,u,f]=e;else if(a==="channelsFirst")[c,f,l,h,u]=e;else throw new Error(`Unknown dataFormat ${a}`);const[g,w,y,,$]=t,[E,_,k]=cn(n),[v,S,T]=cn(s),N=Jt(g,v),D=Jt(w,S),A=Jt(y,T),{padInfo:B,outDepth:C,outHeight:P,outWidth:O}=sl(r,l,h,u,E,_,k,N,D,A,i),K=o?$*f:$;let J;return a==="channelsFirst"?J=[c,K,C,P,O]:a==="channelsLast"&&(J=[c,C,P,O,K]),{batchSize:c,dataFormat:a,inDepth:l,inHeight:h,inWidth:u,inChannels:f,outDepth:C,outHeight:P,outWidth:O,outChannels:K,padInfo:B,strideDepth:E,strideHeight:_,strideWidth:k,filterDepth:g,filterHeight:w,filterWidth:y,effectiveFilterDepth:N,effectiveFilterHeight:D,effectiveFilterWidth:A,dilationDepth:v,dilationHeight:S,dilationWidth:T,inShape:e,outShape:J,filterShape:t}}function tl(e,t,n,s,r){s==null&&(s=js(e,t,n));const o=e[0],a=e[1],i=de((o-t+2*s)/n+1,r),c=de((a-t+2*s)/n+1,r);return[i,c]}function el(e,t,n,s,r,o){r==null&&(r=js(e,t[0],s[0]));const a=[0,0,0,n];for(let i=0;i<3;i++)e[i]+2*r>=t[i]&&(a[i]=de((e[i]-t[i]+2*r)/s[i]+1,o));return a}function js(e,t,n,s=1){const r=Jt(t,s);return Math.floor((e[0]*(n-1)-n+r)/2)}function fe(e){return typeof e=="number"?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function cn(e){return typeof e=="number"?[e,e,e]:e}function Jt(e,t){return t<=1?e:e+(e-1)*(t-1)}function nl(e,t,n,s,r,o,a,i,c){let l,h,u;if(typeof e=="number"){l={top:e,bottom:e,left:e,right:e,type:e===0?"VALID":"NUMBER"};const g=tl([t,n],o,s,e,i);h=g[0],u=g[1]}else if(e==="same"){h=Math.ceil(t/s),u=Math.ceil(n/r);const f=Math.max(0,(h-1)*s+o-t),g=Math.max(0,(u-1)*r+a-n),w=Math.floor(f/2),y=f-w,$=Math.floor(g/2),E=g-$;l={top:w,bottom:y,left:$,right:E,type:"SAME"}}else if(e==="valid")l={top:0,bottom:0,left:0,right:0,type:"VALID"},h=Math.ceil((t-o+1)/s),u=Math.ceil((n-a+1)/r);else if(typeof e=="object"){const f=c==="channelsLast"?e[1][0]:e[2][0],g=c==="channelsLast"?e[1][1]:e[2][1],w=c==="channelsLast"?e[2][0]:e[3][0],y=c==="channelsLast"?e[2][1]:e[3][1];l={top:f,bottom:g,left:w,right:y,type:f===0&&g===0&&w===0&&y===0?"VALID":"EXPLICIT"},h=de((t-o+f+g)/s+1,i),u=de((n-a+w+y)/r+1,i)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outHeight:h,outWidth:u}}function sl(e,t,n,s,r,o,a,i,c,l,h){let u,f,g,w;if(e==="valid"&&(e=0),typeof e=="number"){u={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?"VALID":"NUMBER"};const $=el([t,n,s,1],[i,c,l],1,[r,o,a],e,h);f=$[0],g=$[1],w=$[2]}else if(e==="same"){f=Math.ceil(t/r),g=Math.ceil(n/o),w=Math.ceil(s/a);const y=(f-1)*r+i-t,$=(g-1)*o+c-n,E=(w-1)*a+l-s,_=Math.floor(y/2),k=y-_,v=Math.floor($/2),S=$-v,T=Math.floor(E/2),N=E-T;u={top:v,bottom:S,left:T,right:N,front:_,back:k,type:"SAME"}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:u,outDepth:f,outHeight:g,outWidth:w}}function de(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function De(e){const[t,n,s]=fe(e);return t===1&&n===1&&s===1}function Tt(e,t){return De(e)||De(t)}function Zt(e){return fe(e).every(t=>t>0)}function rl(e){if(e==="NHWC")return"channelsLast";if(e==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function pt(e,t,n){if(n!=null){if(typeof t=="string")throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t=="number")p(ue(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t=="object")t.forEach(s=>{s.forEach(r=>{p(ue(r),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${r}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ol(e,t){const s={x:d(e,"x","reshape","string_or_numeric")},r={shape:t};return b.runKernel(Ra,s,r)}const x=m({reshape_:ol});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function al(e,t,n,s,r){const o=d(e,"x","avgPool","float32"),a=1;p(Tt(n,a),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);let i=o,c=!1;o.rank===3&&(c=!0,i=x(o,[1,o.shape[0],o.shape[1],o.shape[2]])),p(i.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${i.rank}.`),pt("avgPool",s,r);const l={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r};let u=b.runKernel(Xr,l,h);return u=j(u,o.dtype),c?x(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const il=m({avgPool_:al});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cl(e,t,n,s,r,o="NDHWC"){const a=d(e,"x","avgPool3d","float32");let i=a,c=!1;a.rank===4&&(c=!0,i=x(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),p(i.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${i.rank}.`),p(o==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${o}`),p(typeof n=="number"&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),pt("avgPool3d",s,r);const l={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r,dataFormat:o};let u=b.runKernel(Jr,l,h);return u=j(u,i.dtype),c?x(u,[u.shape[1],u.shape[2],u.shape[3],u.shape[4]]):u}const _m=m({avgPool3d_:cl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ll(e,t=0){p(e.length>=1,()=>"Pass at least one tensor to concat");const n=Te(e,"tensors","concat","string_or_numeric");if(n[0].dtype==="complex64"&&n.forEach(o=>{if(o.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${o.dtype}. `)}),n.length===1)return Xt(n[0]);const s=n,r={axis:t};return b.runKernel(ao,s,r)}const rt=m({concat_:ll});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ul(e,t,n=!1,s=!1){let r=d(e,"a","matMul"),o=d(t,"b","matMul");[r,o]=V(r,o);const a={a:r,b:o},i={transposeA:n,transposeB:s};return b.runKernel(Yr,a,i)}const L=m({matMul_:ul});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hl(e){const n={x:d(e,"x","sigmoid","float32")};return b.runKernel(Za,n)}const ce=m({sigmoid_:hl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fl(e,t,n){const s=d(e,"x","slice","string_or_numeric");if(s.rank===0)throw new Error("Slicing scalar is not possible");const r={x:s},o={begin:t,size:n};return b.runKernel(ja,r,o)}const q=m({slice_:fl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dl(e){const n={x:d(e,"x","tanh","float32")};return b.runKernel(wi,n)}const os=m({tanh_:dl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pl(e,t,n,s,r,o){const a=d(e,"forgetBias","basicLSTMCell"),i=d(t,"lstmKernel","basicLSTMCell"),c=d(n,"lstmBias","basicLSTMCell"),l=d(s,"data","basicLSTMCell"),h=d(r,"c","basicLSTMCell"),u=d(o,"h","basicLSTMCell"),f=rt([l,u],1),g=L(f,i),w=M(g,c),y=w.shape[0],$=w.shape[1]/4,E=[y,$],_=q(w,[0,0],E),k=q(w,[0,$],E),v=q(w,[0,$*2],E),S=q(w,[0,$*3],E),T=M(I(ce(_),os(k)),I(h,ce(M(a,v)))),N=I(os(T),ce(S));return[T,N]}const Nm=m({basicLSTMCell_:pl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gl(e,t,n){const s=d(e,"x","batchToSpaceND"),r=t.reduce((i,c)=>i*c);p(s.rank>=1+t.length,()=>`input rank is ${s.rank} but should be > than blockShape.length ${t.length}`),p(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),p(s.shape[0]%r===0,()=>`input tensor batch is ${s.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${r}`);const o={x:s},a={blockShape:t,crops:n};return b.runKernel(Zr,o,a)}const ml=m({batchToSpaceND_:gl});function bl(e){let t;return e.rank===0||e.rank===1?t=x(e,[1,1,1,e.size]):e.rank===2?t=x(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?t=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]):t=e,t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wl(e,t,n,s,r,o){o==null&&(o=.001);const a=d(e,"x","batchNorm"),i=d(t,"mean","batchNorm"),c=d(n,"variance","batchNorm");let l;r!=null&&(l=d(r,"scale","batchNorm"));let h;s!=null&&(h=d(s,"offset","batchNorm")),p(i.rank===c.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),p(h==null||i.rank===h.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),p(l==null||i.rank===l.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const f={x:bl(a),scale:l,offset:h,mean:i,variance:c},g={varianceEpsilon:o},w=b.runKernel(Lo,f,g);return x(w,a.shape)}const En=m({batchNorm_:wl});function yl(e,t,n,s,r,o){const a=d(e,"x","batchNorm"),i=d(t,"mean","batchNorm"),c=d(n,"variance","batchNorm");let l;r!=null&&(l=d(r,"scale","batchNorm"));let h;return s!=null&&(h=d(s,"offset","batchNorm")),p(a.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${a.rank}.`),p(i.rank===2||i.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${i.rank}.`),p(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${c.rank}.`),l!=null&&p(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${l.rank}.`),h!=null&&p(h.rank===2||h.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${h.rank}.`),En(a,i,c,h,l,o)}const Mm=m({batchNorm2d_:yl});function $l(e,t,n,s,r,o){const a=d(e,"x","batchNorm"),i=d(t,"mean","batchNorm"),c=d(n,"variance","batchNorm");let l;r!=null&&(l=d(r,"scale","batchNorm"));let h;return s!=null&&(h=d(s,"offset","batchNorm")),p(a.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${a.rank}.`),p(i.rank===3||i.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${i.rank}.`),p(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${c.rank}.`),l!=null&&p(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${l.rank}.`),h!=null&&p(h.rank===3||h.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${h.rank}.`),En(a,i,c,h,l,o)}const Cm=m({batchNorm3d_:$l});function kl(e,t,n,s,r,o){const a=d(e,"x","batchNorm"),i=d(t,"mean","batchNorm"),c=d(n,"variance","batchNorm");let l;r!=null&&(l=d(r,"scale","batchNorm"));let h;return s!=null&&(h=d(s,"offset","batchNorm")),p(a.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${a.rank}.`),p(i.rank===4||i.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${i.rank}.`),p(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${c.rank}.`),l!=null&&p(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${l.rank}.`),h!=null&&p(h.rank===4||h.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${h.rank}.`),En(a,i,c,h,l,o)}const Bm=m({batchNorm4d_:kl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xl(e,t,n){const s=d(e,"x","bincount"),r=d(t,"weights","bincount");p(s.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${s.dtype}`),p(n>=0,()=>`size must be non-negative, but got ${n}.`),p(r.size===s.size||r.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${s.shape}, weights shape: ${r.shape}.`);const o={x:s,weights:r},a={size:n};return b.runKernel(Qr,o,a)}const El=m({bincount_:xl});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vl(e,t){const n=d(e,"x","bitwiseAnd"),s=d(t,"y","bitwiseAnd");if(!Lt(n.shape,s.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${s.shape}`);if(n.dtype!=="int32"||s.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${s.dtype}`);const r={a:n,b:s};return b.runKernel(to,r)}const Fm=m({bitwiseAnd_:vl});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sl(e,t){const n=d(e,"s0","broadcastArgs","int32"),s=d(t,"s1","broadcastArgs","int32");if(n.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(s.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${s.rank}`);const r={s0:n,s1:s};return b.runKernel(eo,r)}const Rm=m({broadcastArgs_:Sl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Il(e,t){let n=d(e,"broadcastTo","x");const s=n.shape;if(ot(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const l=n.shape.slice();for(;l.length<t.length;)l.unshift(1);n=x(n,l)}const r=n.shape,o=Array.from(t);for(let l=t.length-1;l>=0;l--)if(r[l]===t[l])o[l]=1;else if(n.shape[l]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${t}].`);if(o.map((l,h)=>l>1?h:-1).filter(l=>l>=0).length===0)return Xt(n);const i={x:n},c={reps:o};return b.runKernel(Ss,i,c)}const He=m({broadcastTo_:Il});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tl(e){const n={x:d(e,"x","ceil","float32")};return b.runKernel(no,n)}const Pm=m({ceil_:Tl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oe(e,t,n){ot(e),n=n||Re(t);const s={shape:e,value:t,dtype:n};return b.runKernel(Fo,{},s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dl(e,t,n){const s=d(e,"x","clipByValue");if(p(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return Oe(s.shape,t,s.dtype);const r={x:s},o={clipValueMin:t,clipValueMax:n};return b.runKernel(so,r,o)}const Om=m({clipByValue_:Dl});function Al(e){return rt(e,0)}const Lm=m({concat1d_:Al});function _l(e,t){return rt(e,t)}const Wm=m({concat2d_:_l});function Nl(e,t){return rt(e,t)}const Km=m({concat3d_:Nl});function Ml(e,t){return rt(e,t)}const Um=m({concat4d_:Ml});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cl(e,t,n,s,r="NHWC",o=[1,1],a){const i=d(e,"x","conv2d","float32"),c=d(t,"filter","conv2d","float32");let l=i,h=!1;i.rank===3&&(h=!0,l=x(i,[1,i.shape[0],i.shape[1],i.shape[2]])),p(l.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),p(c.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),pt("conv2d",s,a);const u=r==="NHWC"?l.shape[3]:l.shape[1];p(u===c.shape[2],()=>`Error in conv2d: depth of input (${u}) must match input depth for filter ${c.shape[2]}.`),p(Tt(n,o),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),p(Zt(o),()=>"Error in conv2D: Dilated rates should be larger than 0."),p(Zt(n),()=>"Error in conv2D: Strides should be larger than 0.");const f={x:l,filter:c},g={strides:n,pad:s,dataFormat:r,dilations:o,dimRoundingMode:a},w=b.runKernel(io,f,g);return h?x(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const vn=m({conv2d_:Cl});function Bl(e,t,n,s,r="NWC",o=1,a){const i=d(e,"x","conv1d"),c=d(t,"filter","conv1d");let l=i,h=!1;i.rank===2&&(h=!0,l=x(i,[1,i.shape[0],i.shape[1]])),p(l.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${l.rank}.`),p(c.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${c.rank}.`),pt("conv1d",s,a),p(l.shape[2]===c.shape[1],()=>`Error in conv1d: depth of input (${l.shape[2]}) must match input depth for filter ${c.shape[1]}.`),p(Tt(n,o),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${o}'`),p(Zt(o),()=>"Error in conv1D: Dilated rates should be larger than 0."),p(Zt(n),()=>"Error in conv1D: Stride should be larger than 0."),p(r==="NWC",()=>`Error in conv1d: got dataFormat of ${r} but only NWC is currently supported.`);const u=x(c,[1,c.shape[0],c.shape[1],c.shape[2]]),f=x(l,[l.shape[0],1,l.shape[1],l.shape[2]]),$=vn(f,u,[1,n],s,"NHWC",[1,o],a);return h?x($,[$.shape[2],$.shape[3]]):x($,[$.shape[0],$.shape[2],$.shape[3]])}const Gm=m({conv1d_:Bl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fl(e,t,n,s,r,o="NHWC",a){p(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let i=e,c=t,l=!1;t.rank===3&&(l=!0,c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]),i=[1,e[0],e[1],e[2]]),p(i.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${i.length}.`),p(c.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${c.rank}`),p(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const h=o==="NHWC"?i[3]:i[1],u=o==="NHWC"?c.shape[3]:c.shape[1];p(h===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${h}) must match input depth for filter ${n.shape[2]}.`),p(u===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${u}) must match output depth for filter ${n.shape[3]}.`),pt("conv2dDerInput",r,a);const f={dy:c,filter:n},g={strides:s,pad:r,dataFormat:o,dimRoundingMode:a,inputShape:i},w=b.runKernel(lo,f,g);return l?x(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const Xs=m({conv2DBackpropInput_:Fl});function Rl(e,t,n,s,r,o){const a=d(e,"x","conv2dTranspose"),i=d(t,"filter","conv2dTranspose");return Xs(n,a,i,s,r,"NHWC",o)}const zm=m({conv2dTranspose_:Rl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pl(e,t,n,s,r="NDHWC",o=[1,1,1]){const a=d(e,"x","conv3d"),i=d(t,"filter","conv3d");let c=a,l=!1;a.rank===4&&(l=!0,c=x(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),p(c.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${c.rank}.`),p(i.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${i.rank}.`),p(c.shape[4]===i.shape[3],()=>`Error in conv3d: depth of input (${c.shape[4]}) must match input depth for filter ${i.shape[3]}.`),p(Tt(n,o),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),p(r==="NDHWC",()=>`Error in conv3d: got dataFormat of ${r} but only NDHWC is currently supported.`),p(Zt(o),()=>"Error in conv3D: Dilated rates should be larger than 0."),p(Zt(n),()=>"Error in conv3D: Strides should be larger than 0.");const h={x:c,filter:i},u={strides:n,pad:s,dataFormat:r,dilations:o},f=b.runKernel(uo,h,u);return l?x(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}const qm=m({conv3d_:Pl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ol(e,t,n,s,r){p(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let o=e,a=t,i=!1;t.rank===4&&(i=!0,a=x(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),o=[1,e[0],e[1],e[2],e[3]]);const c=o[4],l=a.shape[4];p(o.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${o.length}.`),p(a.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${a.rank}`),p(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),p(c===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${c}) must match input depth for filter ${n.shape[3]}.`),p(l===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[4]}.`);const h={dy:a,filter:n},u={pad:r,strides:s,inputShape:o},f=b.runKernel(ho,h,u);return i?x(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}const Ll=m({conv3DBackpropInput_:Ol});function Wl(e,t,n,s,r){const o=d(e,"x","conv3dTranspose"),a=d(t,"filter","conv3dTranspose");return Ll(n,o,a,s,r)}const Hm=m({conv3dTranspose_:Wl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kl(e){const n={x:d(e,"x","cos","float32")};return b.runKernel(fo,n)}const Vm=m({cos_:Kl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ul(e){const n={x:d(e,"x","cosh","float32")};return b.runKernel(po,n)}const jm=m({cosh_:Ul});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gl(e,t=0,n=!1,s=!1){const o={x:d(e,"x","cumprod")},a={axis:t,exclusive:n,reverse:s};return b.runKernel(go,o,a)}const Xm=m({cumprod_:Gl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zl(e,t=0,n=!1,s=!1){const o={x:d(e,"x","cumsum")},a={axis:t,exclusive:n,reverse:s};return b.runKernel(mo,o,a)}const Jm=m({cumsum_:zl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ql(e,t,n,s=!1){const r=d(e,"x","denseBincount"),o=d(t,"weights","denseBincount");p(r.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${r.dtype}`),p(r.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${r.rank}.`),p(n>=0,()=>`size must be non-negative, but got ${n}.`),p(o.size===r.size||o.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${r.shape}, weights shape: ${o.shape}.`);const a={x:r,weights:o},i={size:n,binaryOutput:s};return b.runKernel(wo,a,i)}const Ym=m({denseBincount_:ql});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hl(e,t,n="NHWC"){const s=d(e,"x","depthToSpace","float32"),r=n==="NHWC"?s.shape[1]:s.shape[2],o=n==="NHWC"?s.shape[2]:s.shape[3],a=n==="NHWC"?s.shape[3]:s.shape[1];p(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),p(r*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${r} and ${t}  for depthToSpace with input shape
    ${s.shape}`),p(o*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${o} and ${t} for depthToSpace with input shape
        ${s.shape}`),p(a%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${a} for depthToSpace with input shape ${s.shape}`);const i={x:s},c={blockSize:t,dataFormat:n};return b.runKernel(yo,i,c)}const Zm=m({depthToSpace_:Hl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vl(e,t,n,s,r="NHWC",o=[1,1],a){const i=d(e,"x","depthwiseConv2d","float32"),c=d(t,"filter","depthwiseConv2d","float32");let l=i,h=!1;i.rank===3&&(h=!0,l=x(i,[1,i.shape[0],i.shape[1],i.shape[2]])),p(l.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${l.rank}.`),p(c.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${c.rank}.`);const u=r==="NHWC"?l.shape[3]:l.shape[1];p(u===c.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${u}) must match the inChannels dimension in filter ${c.shape[2]}.`),pt("depthwiseConv2d",s,a);const f={x:l,filter:c},g={strides:n,pad:s,dataFormat:r,dilations:o,dimRoundingMode:a},w=b.runKernel($o,f,g);return h?x(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const Js=m({depthwiseConv2d_:Vl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jl(e){const n={x:d(e,"x","diag")};return b.runKernel(Eo,n)}const Qm=m({diag_:jl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xl(e,t,n,s,r=[1,1],o="NHWC"){const a=d(e,"x","dilation2d"),i=d(t,"filter","dilation2d");p(a.rank===3||a.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${a.rank}.`),p(i.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${i.rank}.`),p(o==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${o}`);let c=a,l=!1;a.rank===3&&(c=x(a,[1,a.shape[0],a.shape[1],a.shape[2]]),l=!0),p(c.shape[3]===i.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${c.shape[3]} vs ${i.shape[2]}`);const h={x:c,filter:i},u={strides:n,pad:s,dilations:r},f=b.runKernel(vo,h,u);return l?x(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const tb=m({dilation2d_:Xl});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eb(e,t){const n=e.length,s=[];for(let r=0;r<n;r++){const o=n-1-r,a=e[o]||1;(t[t.length-1-r]||1)>1&&a===1&&s.unshift(o)}return s}function Jl(e,t){const n=[];for(let s=0;s<t.length;s++){const r=e[e.length-s-1],o=t.length-s-1,a=t[o];(r==null||r===1&&a>1)&&n.unshift(o)}return n}function Y(e,t){const n=Math.max(e.length,t.length),s=new Array(n);for(let r=0;r<n;r++){let o=e[e.length-r-1];o==null&&(o=1);let a=t[t.length-r-1];if(a==null&&(a=1),o===1)s[n-r-1]=a;else if(a===1)s[n-r-1]=o;else if(o!==a){const i=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(i)}else s[n-r-1]=o}return s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yl(e,t){let n=d(e,"a","equal","string_or_numeric"),s=d(t,"b","equal","string_or_numeric");[n,s]=V(n,s),Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(_o,r)}const Zl=m({equal_:Yl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ql(e,t,n){const s=d(t,"a","where"),r=d(n,"b","where"),o=d(e,"condition","where","bool"),a=Y(Y(o.shape,s.shape),r.shape),i=He(o,a),c=He(s,a),l=He(r,a),h={condition:i,t:c,e:l};return b.runKernel(Ha,h)}const Mt=m({where_:Ql});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tu(e){const n={x:d(e,"x","zerosLike")};return b.runKernel(vi,n)}const ct=m({zerosLike_:tu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eu(e,t){let n=d(e,"a","div"),s=d(t,"b","div");[n,s]=V(n,s);const r=G(n,s),o=ct(r),a=Zl(s,o);return Mt(a,o,r)}const nb=m({divNoNan_:eu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nu(e,t){const n=d(e,"t1","dot"),s=d(t,"t2","dot");p((n.rank===1||n.rank===2)&&(s.rank===1||s.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${s.rank}.`);const r=n.rank===1?n.size:n.shape[1],o=s.rank===1?s.size:s.shape[0];if(p(r===o,()=>`Error in dot: inner dimensions of inputs must match, but got ${r} and ${o}.`),n.rank===1&&s.rank===1){const a=x(n,[1,-1]),i=x(s,[-1,1]),c=L(a,i);return x(c,[])}else if(n.rank===1&&s.rank===2){const a=x(n,[1,-1]),i=x(s,[s.shape[0],s.shape[1]]),c=L(a,i);return x(c,[c.size])}else if(n.rank===2&&s.rank===1){const a=x(s,[-1,1]),i=L(n,a);return x(i,[i.size])}else{const a=x(s,[s.shape[0],s.shape[1]]);return L(n,a)}}const sb=m({dot_:nu});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function su(e,...t){const n=t.map((r,o)=>d(r,`tensors${o}`,"einsum")),s={equation:e};return b.runKernel(To,n,s)}const re=m({einsum_:su});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ru(e){const n={x:d(e,"x","elu","float32")};return b.runKernel(Do,n)}const ou=m({elu_:ru});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function au(e,t){const n=d(e,"x","ensureShape","string_or_numeric");if(!xr(n.shape,t))throw new Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${t}`);return e}const rb=m({ensureShape_:au});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iu(e){let t=d(e,"x","erf");p(t.dtype==="int32"||t.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),t.dtype==="int32"&&(t=j(t,"float32"));const n={x:t};return b.runKernel(Ao,n)}const ob=m({erf_:iu});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ys(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function cu(e,t,n){const s=e.length+t.length,r=[];let o=0,a=0;for(let i=0;i<s;i++)n.indexOf(i)===-1?r.push(e[o++]):r.push(t[a++]);return r}function ab(e,t){const n=[],s=e.length;for(let o=0;o<s;o++)t.indexOf(o)===-1&&n.push(e[o]);const r=t.map(o=>e[o]);return[n,r]}function Le(e,t){const n=t.map(s=>1);return cu(e,n,t)}function ib(e,t,n){p(Ys(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function cb(e,t){if(Ys(e,t))return null;const n=[];for(let s=0;s<t;++s)e.indexOf(s)===-1&&n.push(s);return e.forEach(s=>n.push(s)),n}function lb(e){return e.map((t,n)=>[n,t]).sort((t,n)=>t[1]-n[1]).map(t=>t[0])}function ub(e,t){const n=[];for(let s=t-e;s<t;++s)n.push(s);return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lu(e,t=null,n=!1){const r={x:d(e,"x","max")},o={reductionIndices:t,keepDims:n};return b.runKernel(oa,r,o)}const le=m({max_:lu});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uu(e,t=null,n=!1){const r={x:d(e,"x","min")},o={axis:t,keepDims:n};return b.runKernel(ha,r,o)}const as=m({min_:uu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hu(e,t){let n=d(e,"base","pow"),s=d(t,"exp","pow");[n,s]=V(n,s);const r={a:n,b:s};return b.runKernel(Ia,r)}const Ae=m({pow_:hu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function W(e,t){if((lt(e)&&t!=="string"||Array.isArray(e))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&lt(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return It(e,[],[],t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fu(e){const n={x:d(e,"x","sqrt","float32")};return b.runKernel(ti,n)}const vt=m({sqrt_:fu});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function du(e){const t=d(e,"x","square"),n={};return b.runKernel("Square",{x:t},n)}const gt=m({square_:du});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pu(e,t=null,n=!1){let s=d(e,"x","sum");s.dtype==="bool"&&(s=j(s,"int32"));const r={x:s},o={axis:t,keepDims:n};return b.runKernel(ei,r,o)}const U=m({sum_:pu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gu(e,t="euclidean",n=null,s=!1){e=d(e,"x","norm");const r=Zs(e,t,n);let o=r.shape;if(s){const a=Fe(n,e.shape);o=Le(r.shape,a)}return x(r,o)}function Zs(e,t,n=null){if(e.rank===0)return it(e);if(e.rank!==1&&n===null)return Zs(x(e,[-1]),t,n);if(e.rank===1||typeof n=="number"||Array.isArray(n)&&n.length===1){if(t===1)return U(it(e),n);if(t===1/0)return le(it(e),n);if(t===-1/0)return as(it(e),n);if(t==="euclidean"||t===2)return vt(U(Ae(it(e),W(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return le(U(it(e),n[0]),n[1]-1);if(t===1/0)return le(U(it(e),n[1]),n[0]);if(t===-1/0)return as(U(it(e),n[1]),n[0]);if(t==="fro"||t==="euclidean")return vt(U(gt(e),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const Sn=m({norm_:gu});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mu(e,t=null,n=!1){return Sn(e,"euclidean",t,n)}const hb=m({euclideanNorm_:mu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bu(e){const n={x:d(e,"x","exp")};return b.runKernel(No,n)}const Qt=m({exp_:bu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wu(e,t=0){const n=d(e,"x","expandDims","string_or_numeric");p(t<=n.rank,()=>"Axis must be <= rank of the tensor");const s={input:n},r={dim:t};return b.runKernel(Mo,s,r)}const At=m({expandDims_:wu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yu(e){const n={x:d(e,"x","expm1")};return b.runKernel(Co,n)}const fb=m({expm1_:yu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $u(e,t){const n=d(e,"x","tile","string_or_numeric");p(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);const s={x:n},r={reps:t};return b.runKernel(Ss,s,r)}const ke=m({tile_:$u});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ku(e,t,n,s="float32"){t==null&&(t=e);const r=Et([e,t],s),o=e<=t?e:t;for(let i=0;i<o;++i)r.set(1,i,i);const a=x(r.toTensor(),[e,t]);if(n==null)return a;if(n.length===1)return ke(At(a,0),[n[0],1,1]);if(n.length===2)return ke(At(At(a,0),0),[n[0],n[1],1,1]);if(n.length===3)return ke(At(At(At(a,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}const xu=m({eye_:ku});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eu(e){const n={x:d(e,"x","floor","float32")};return b.runKernel(Po,n)}const vu=m({floor_:Eu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Su(e,t,n=0,s=0){const r=d(e,"x","gather"),o=d(t,"indices","gather","int32"),a={x:r,indices:o},i={axis:n,batchDims:s};return b.runKernel(Wo,a,i)}const Iu=m({gather_:Su});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tu(e,t){let n=d(e,"a","greater","string_or_numeric"),s=d(t,"b","greater","string_or_numeric");[n,s]=V(n,s),Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Uo,r)}const In=m({greater_:Tu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Du(e,t){let n=d(e,"a","greaterEqual","string_or_numeric"),s=d(t,"b","greaterEqual","string_or_numeric");[n,s]=V(n,s),Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Go,r)}const Au=m({greaterEqual_:Du});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _u(e){const n={input:d(e,"input","imag")};return b.runKernel(qo,n)}const Tn=m({imag_:_u});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nu(e){const n={x:d(e,"x","isFinite")};return b.runKernel(Ho,n)}const db=m({isFinite_:Nu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mu(e){const n={x:d(e,"x","isInf")};return b.runKernel(Vo,n)}const pb=m({isInf_:Mu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cu(e){const n={x:d(e,"x","isNaN")};return b.runKernel(jo,n)}const gb=m({isNaN_:Cu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bu(e,t=.2){const s={x:d(e,"x","leakyRelu")},r={alpha:t};return b.runKernel(Xo,s,r)}const Fu=m({leakyRelu_:Bu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ru(e,t){let n=d(e,"a","less","string_or_numeric"),s=d(t,"b","less","string_or_numeric");[n,s]=V(n,s),Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Jo,r)}const is=m({less_:Ru});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pu(e,t){let n=d(e,"a","lessEqual","string_or_numeric"),s=d(t,"b","lessEqual","string_or_numeric");[n,s]=V(n,s),Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Yo,r)}const Qs=m({lessEqual_:Pu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mb(e,t,n){if(n<=0)throw new Error("The number of values should be positive.");const s={start:e,stop:t,num:n};return b.runKernel(Zo,{},s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ou(e,t=5,n=1,s=1,r=.5){const o=d(e,"x","localResponseNormalization");p(o.rank===4||o.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${o.rank}.`),p(ue(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let a=o,i=!1;o.rank===3&&(i=!0,a=x(o,[1,o.shape[0],o.shape[1],o.shape[2]]));const c={x:a},l={depthRadius:t,bias:n,alpha:s,beta:r},h=b.runKernel(ra,c,l);return i?x(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const bb=m({localResponseNormalization_:Ou});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lu(e){const n={x:d(e,"x","log","float32")};return b.runKernel(Qo,n)}const _e=m({log_:Lu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wu(e){const n={x:d(e,"x","log1p")};return b.runKernel(ta,n)}const Ku=m({log1p_:Wu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uu(e,t){p(Je(e),()=>"The f passed in variableGrads(f) must be a function"),p(t==null||Array.isArray(t)&&t.every(l=>l instanceof Ie),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=t!=null;if(!n){t=[];for(const l in b.registeredVariables)t.push(b.registeredVariables[l])}const s=n?t.filter(l=>!l.trainable):null,r=t.length;t=t.filter(l=>l.trainable),p(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const o=!0,{value:a,grads:i}=b.gradients(e,t,null,o);p(i.some(l=>l!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),p(a.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${a.rank} tensor`);const c={};return t.forEach((l,h)=>{i[h]!=null&&(c[l.name]=i[h])}),s!=null&&s.forEach(l=>c[l.name]=null),{value:a,grads:c}}function yt(e){return b.customGrad(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gu(e){const n={x:d(e,"x","neg")};return b.runKernel(ba,n)}const wt=m({neg_:Gu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zu(e){const n={x:d(e,"x","softplus")};return b.runKernel(Qa,n)}const qu=m({softplus_:zu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hu(e){const t=d(e,"x","logSigmoid");return yt(s=>({value:wt(qu(wt(s))),gradFunc:a=>I(a,ce(wt(s)))}))(t)}const wb=m({logSigmoid_:Hu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vu(e,t){let n=d(e,"a","sub"),s=d(t,"b","sub");[n,s]=V(n,s);const r={a:n,b:s};return b.runKernel(mi,r)}const F=m({sub_:Vu});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ju(e,t=-1){const n=d(e,"logits","logSoftmax");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return yt((r,o)=>{const i=le(r,t,!0),c=F(r,i),l=F(j(c,"float32"),_e(U(Qt(c),t,!0)));return o([l]),{value:l,gradFunc:(u,f)=>{const[g]=f,w=!0,y=Qt(g);return F(u,I(U(u,t,w),y))}}})(n)}const yb=m({logSoftmax_:ju});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xu(e,t=null,n=!1){const s=d(e,"x","logSumExp"),r=Fe(t,s.shape),o=le(s,r,!0),a=F(s,o),i=Qt(a),c=U(i,r),l=_e(c),h=M(x(o,l.shape),l);if(n){const u=Le(h.shape,r);return x(h,u)}return h}const Ju=m({logSumExp_:Xu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yu(e,t){const n=d(e,"a","logicalAnd","bool"),s=d(t,"b","logicalAnd","bool");Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(ea,r)}const ln=m({logicalAnd_:Yu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zu(e){const n={x:d(e,"x","logicalNot","bool")};return b.runKernel(na,n)}const Qu=m({logicalNot_:Zu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function th(e,t){const n=d(e,"a","logicalOr","bool"),s=d(t,"b","logicalOr","bool");Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(sa,r)}const eh=m({logicalOr_:th});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nh(e,t){const n=d(e,"a","logicalXor","bool"),s=d(t,"b","logicalXor","bool");return Y(n.shape,s.shape),ln(eh(e,t),Qu(ln(e,t)))}const $b=m({logicalXor_:nh});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const be=2147483648;function sh(e,t,n="left"){const s=d(e,"sortedSequence","searchSorted"),r=d(t,"values","searchSorted"),o=s.shape[s.shape.length-1],a=r.shape[r.shape.length-1],i=x(s,[-1,o]),c=x(r,[-1,a]);if(i.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(i.shape[0]!==c.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(H(c.shape)>=be)throw new Error(`values tensor size must less than ${be}`);if(i.shape[1]>=be)throw new Error(`trailing dim_size must less than ${be} for int32 output type, was ${i.shape[1]}`);const l={sortedSequence:i,values:c},h={side:n};return b.runKernel(qa,l,h)}const tr=m({searchSorted_:sh});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kb(e,t){return tr(e,t,"left")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rh(e,t,n,s,r){const o=d(e,"x","maxPool"),a=1;let i=o,c=!1;o.rank===3&&(c=!0,i=x(o,[1,o.shape[0],o.shape[1],o.shape[2]])),p(i.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${i.rank}.`),p(Tt(n,a),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),pt("maxPool",s,r);const l={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r},u=b.runKernel(ia,l,h);return c?x(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const oh=m({maxPool_:rh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ah(e,t=[1,1,1],n,s,r,o="NDHWC"){const a=d(e,"x","maxPool3d");let i=a,c=!1;a.rank===4&&(c=!0,i=x(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),p(i.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${i.rank}.`),p(o==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${o}`),pt("maxPool3d",s,r);const l={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r,dataFormat:o},u=b.runKernel(ca,l,h);return c?x(u,[u.shape[1],u.shape[2],u.shape[3],u.shape[4]]):u}const xb=m({maxPool3d_:ah});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ih(e,t,n,s,r=!1){const a={x:d(e,"x","maxPoolWithArgmax")},i={filterSize:t,strides:n,pad:s,includeBatchInIndex:r},c=b.runKernel(la,a,i);return{result:c[0],indexes:c[1]}}const Eb=m({maxPoolWithArgmax_:ih});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ch(e,t){let n=d(e,"a","maximum"),s=d(t,"b","maximum");[n,s]=V(n,s),n.dtype==="bool"&&(n=j(n,"int32"),s=j(s,"int32")),Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(aa,r)}const lh=m({maximum_:ch});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uh(e,t=null,n=!1){const r={x:d(e,"x","mean")},o={axis:t,keepDims:n};return b.runKernel(ua,r,o)}const un=m({mean_:uh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pe(e,t="float32"){if(ot(e),t==="complex64"){const s=pe(e,"float32"),r=pe(e,"float32");return Ct(s,r)}const n=mn(H(e),t);return b.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vt(e,t="float32"){if(ot(e),t==="complex64"){const s=Vt(e,"float32"),r=pe(e,"float32");return Ct(s,r)}const n=ys(H(e),t);return b.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vb(e,t,{indexing:n="xy"}={}){if(n!=="xy"&&n!=="ij")throw new TypeError(`${n} is not a valid third argument to meshgrid`);if(e===void 0)return[];let s=d(e,"x","meshgrid",e instanceof Q?e.dtype:"float32");if(t===void 0)return[s];let r=d(t,"y","meshgrid",t instanceof Q?t.dtype:"float32");const o=H(s.shape),a=H(r.shape);return n==="xy"?(s=x(s,[1,-1]),r=x(r,[-1,1]),[L(Vt([a,1],s.dtype),s),L(r,Vt([1,o],r.dtype))]):(s=x(s,[-1,1]),r=x(r,[1,-1]),[L(s,Vt([1,a],s.dtype)),L(Vt([o,1],r.dtype),r)])}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hh(e,t){let n=d(e,"a","minimum"),s=d(t,"b","minimum");[n,s]=V(n,s),n.dtype==="bool"&&(n=j(n,"int32"),s=j(s,"int32")),Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(fa,r)}const hn=m({minimum_:hh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fh(e,t,n){p(n==="reflect"||n==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);const s=d(e,"x","mirrorPad");if(s.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");p(t.length===s.rank,()=>`Padding doesn't match input. Must be ${s.rank}. Got ${t.length}.`);const r=n==="reflect"?1:0;for(let i=0;i<s.rank;i++)p(t[i].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),p(t[i][0]>=0&&t[i][0]<=s.shape[i]-r&&t[i][1]>=0&&t[i][1]<=s.shape[i]-r,()=>`Padding in dimension ${i} cannot be greater than or equal to ${s.shape[i]-r} or less than 0 for input of shape ${s.shape}`);const o={paddings:t,mode:n},a={x:s};return b.runKernel(da,a,o)}const Sb=m({mirrorPad_:fh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dh(e,t){let n=d(e,"a","mod"),s=d(t,"b","mod");[n,s]=V(n,s);const r={a:n,b:s};return b.runKernel(pa,r)}const Ib=m({mod_:dh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ph(e,t=null,n=!1){e=d(e,"x","moments");const s=Fe(t,e.shape),r=un(e,s,n);let o=r.shape;n||(o=Le(r.shape,s));const a=gt(F(j(e,"float32"),x(r,o))),i=un(a,s,n);return{mean:r,variance:i}}const Tb=m({moments_:ph});function gh(e,t,n,s){const r=d(t,"data","multiRNNCell"),o=Te(n,"c","multiRNNCell"),a=Te(s,"h","multiRNNCell");let i=r;const c=[];for(let u=0;u<e.length;u++){const f=e[u](i,o[u],a[u]);c.push(f[0]),c.push(f[1]),i=f[1]}const l=[],h=[];for(let u=0;u<c.length;u+=2)l.push(c[u]),h.push(c[u+1]);return[l,h]}const Db=m({multiRNNCell_:gh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mh(e,t,n,s=!1){const r=d(e,"logits","multinomial"),o=r.size,a=r.rank;if(o<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${o}.`);if(a>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${a}`);n=n||Math.random();const c={logits:a===1?x(r,[1,-1]):r},l={numSamples:t,seed:n,normalized:s},h=b.runKernel(ga,c,l);return a===1?x(h,[h.size]):h}const Ab=m({multinomial_:mh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bh(e,t){let n=d(e,"a","notEqual","string_or_numeric"),s=d(t,"b","notEqual","string_or_numeric");[n,s]=V(n,s),Y(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(wa,r)}const wh=m({notEqual_:bh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yh(e,t,n=1,s=0,r="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const a={indices:d(e,"indices","oneHot","int32")},i={dtype:r,depth:t,onValue:n,offValue:s};return b.runKernel(Ea,a,i)}const _b=m({oneHot_:yh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $h(e){const n={x:d(e,"x","onesLike")};return b.runKernel(xa,n)}const Nb=m({onesLike_:$h});function kh(e,t){const n=d(e,"v1","outerProduct"),s=d(t,"v2","outerProduct");p(n.rank===1&&s.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${s.rank}.`);const r=x(n,[-1,1]),o=x(s,[1,-1]);return L(r,o)}const Mb=m({outerProduct_:kh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xh(e,t,n=0){const s=d(e,"x","pad");if(s.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const r={paddings:t,constantValue:n},o={x:s};return b.runKernel(Sa,o,r)}const We=m({pad_:xh});function Eh(e,t,n=0){return p(t.length===2,()=>"Invalid number of paddings. Must be length of 2."),We(e,[t],n)}const Cb=m({pad1d_:Eh});function vh(e,t,n=0){return p(t.length===2&&t[0].length===2&&t[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),We(e,t,n)}const Bb=m({pad2d_:vh});function Sh(e,t,n=0){return p(t.length===3&&t[0].length===2&&t[1].length===2&&t[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),We(e,t,n)}const Fb=m({pad3d_:Sh});function Ih(e,t,n=0){return p(t.length===4&&t[0].length===2&&t[1].length===2&&t[2].length===2&&t[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),We(e,t,n)}const Rb=m({pad4d_:Ih});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Th(e,t,n){const s=d(e,"x","spaceToBatchND");p(s.rank>=1+t.length,()=>`input rank ${s.rank} should be > than [blockShape] ${t.length}`),p(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),p(s.shape.reduce((a,i,c)=>c>0&&c<=t.length?a&&(i+n[c-1][0]+n[c-1][1])%t[c-1]===0:a,!0),()=>`input spatial dimensions ${s.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);const r={x:s},o={blockShape:t,paddings:n};return b.runKernel(ni,r,o)}const Dh=m({spaceToBatchND_:Th});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ah(e,t,n,s,r,o,a){r==null&&(r=[1,1]),o==null&&(o=1),s===0&&(s="valid");const i=d(e,"x","maxPool");let c=i,l=!1;i.rank===3&&(l=!0,c=x(i,[1,i.shape[0],i.shape[1],i.shape[2]])),p(Tt(o,r),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${o} and dilations '${r}'`);const h=Zc(c.shape,t,o,r,s),u=[h.dilationHeight,h.dilationWidth];let f;s==="same"?f=Nh([h.filterHeight,h.filterWidth],u):f=[[0,0],[0,0]];const g=u[0]===1&&u[1]===1,[w,y]=_h([h.inHeight,h.inWidth],u,f),$=g?s:"valid",E=g?c:Dh(c,u,w),k=(n==="avg"?()=>il(E,t,o,$,a):()=>oh(E,t,o,$,a))(),v=g?k:ml(k,u,y);return l?x(v,[v.shape[1],v.shape[2],v.shape[3]]):v}function _h(e,t,n){const s=n.map(h=>h[0]),r=n.map(h=>h[1]),o=e.concat(s,r),a=t.map((h,u)=>(h-o[u]%h)%h),i=r.map((h,u)=>h+a[u]),c=t.map((h,u)=>[s[u],i[u]]),l=t.map((h,u)=>[0,a[u]]);return[c,l]}function Nh(e,t){const s=e.map((a,i)=>a+(a-1)*(t[i]-1)).map(a=>a-1),r=s.map(a=>Math.floor(a/2)),o=s.map((a,i)=>a-r[i]);return s.map((a,i)=>[r[i],o[i]])}const Pb=m({pool_:Ah});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mh(e,t){const n=d(e,"x","prelu"),s=d(t,"alpha","prelu"),r={x:n,alpha:s};return b.runKernel(Ta,r)}const Ch=m({prelu_:Mh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bh(e,t=null,n=!1){let s=d(e,"x","prod");s.dtype==="bool"&&(s=j(s,"int32"));const r={x:s},o={axis:t,keepDims:n};return b.runKernel(Da,r,o)}const Ob=m({prod_:Bh});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fh(e,t,n,s){const r=e.map((h,u)=>d(h,`tensors${u}`,"raggedGather","int32")),o=d(t,"paramsDenseValues","raggedGather"),a=d(n,"indices","raggedGather","int32"),i={paramsNestedSplits:r,paramsDenseValues:o,indices:a},c={outputRaggedRank:s},l=b.runKernel(Aa,i,c);return{outputNestedSplits:l.slice(0,l.length-1),outputDenseValues:l[l.length-1]}}const Lb=m({raggedGather_:Fh});/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rh(e,t,n){const s=d(e,"starts","raggedRange"),r=d(t,"limits","raggedRange",s.dtype),o=d(n,"deltas","raggedRange",s.dtype),a={starts:s,limits:r,deltas:o},i=b.runKernel(_a,a);return{rtNestedSplits:i[0],rtDenseValues:i[1]}}const Wb=m({raggedRange_:Rh});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ph(e,t,n,s,r){const o=d(e,"shape","raggedTensorToTensor","int32"),a=d(t,"values","raggedTensorToTensor"),i=d(n,"defaultValue","raggedTensorToTensor",a.dtype),c=s.map((u,f)=>d(u,`tensors${f}`,"raggedTensorToTensor","int32")),l={shape:o,values:a,defaultValue:i,rowPartitionTensors:c},h={rowPartitionTypes:r};return b.runKernel(Na,l,h)}const Kb=m({raggedTensorToTensor_:Ph});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oh(e,t,n){ot(e);const s=H(e);let r=null;if(n==null||n==="float32")r=new Float32Array(s);else if(n==="int32")r=new Int32Array(s);else if(n==="bool")r=new Uint8Array(s);else throw new Error(`Unknown data type ${n}`);for(let o=0;o<s;o++)r[o]=t();return b.makeTensor(r,e,n)}const Ub=m({rand_:Oh});var Dn={exports:{}};Dn.exports;(function(e){(function(t,n,s){function r(c){var l=this,h=i();l.next=function(){var u=2091639*l.s0+l.c*23283064365386963e-26;return l.s0=l.s1,l.s1=l.s2,l.s2=u-(l.c=u|0)},l.c=1,l.s0=h(" "),l.s1=h(" "),l.s2=h(" "),l.s0-=h(c),l.s0<0&&(l.s0+=1),l.s1-=h(c),l.s1<0&&(l.s1+=1),l.s2-=h(c),l.s2<0&&(l.s2+=1),h=null}function o(c,l){return l.c=c.c,l.s0=c.s0,l.s1=c.s1,l.s2=c.s2,l}function a(c,l){var h=new r(c),u=l&&l.state,f=h.next;return f.int32=function(){return h.next()*4294967296|0},f.double=function(){return f()+(f()*2097152|0)*11102230246251565e-32},f.quick=f,u&&(typeof u=="object"&&o(u,h),f.state=function(){return o(h,{})}),f}function i(){var c=4022871197,l=function(h){h=String(h);for(var u=0;u<h.length;u++){c+=h.charCodeAt(u);var f=.02519603282416938*c;c=f>>>0,f-=c,f*=c,c=f>>>0,f-=c,c+=f*4294967296}return(c>>>0)*23283064365386963e-26};return l}n&&n.exports?n.exports=a:this.alea=a})(Ot,e)})(Dn);var Lh=Dn.exports,An={exports:{}};An.exports;(function(e){(function(t,n,s){function r(i){var c=this,l="";c.x=0,c.y=0,c.z=0,c.w=0,c.next=function(){var u=c.x^c.x<<11;return c.x=c.y,c.y=c.z,c.z=c.w,c.w^=c.w>>>19^u^u>>>8},i===(i|0)?c.x=i:l+=i;for(var h=0;h<l.length+64;h++)c.x^=l.charCodeAt(h)|0,c.next()}function o(i,c){return c.x=i.x,c.y=i.y,c.z=i.z,c.w=i.w,c}function a(i,c){var l=new r(i),h=c&&c.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,g=(l.next()>>>0)/4294967296,w=(f+g)/(1<<21);while(w===0);return w},u.int32=l.next,u.quick=u,h&&(typeof h=="object"&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=a:this.xor128=a})(Ot,e)})(An);var Wh=An.exports,_n={exports:{}};_n.exports;(function(e){(function(t,n,s){function r(i){var c=this,l="";c.next=function(){var u=c.x^c.x>>>2;return c.x=c.y,c.y=c.z,c.z=c.w,c.w=c.v,(c.d=c.d+362437|0)+(c.v=c.v^c.v<<4^(u^u<<1))|0},c.x=0,c.y=0,c.z=0,c.w=0,c.v=0,i===(i|0)?c.x=i:l+=i;for(var h=0;h<l.length+64;h++)c.x^=l.charCodeAt(h)|0,h==l.length&&(c.d=c.x<<10^c.x>>>4),c.next()}function o(i,c){return c.x=i.x,c.y=i.y,c.z=i.z,c.w=i.w,c.v=i.v,c.d=i.d,c}function a(i,c){var l=new r(i),h=c&&c.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,g=(l.next()>>>0)/4294967296,w=(f+g)/(1<<21);while(w===0);return w},u.int32=l.next,u.quick=u,h&&(typeof h=="object"&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=a:this.xorwow=a})(Ot,e)})(_n);var Kh=_n.exports,Nn={exports:{}};Nn.exports;(function(e){(function(t,n,s){function r(i){var c=this;c.next=function(){var h=c.x,u=c.i,f,g;return f=h[u],f^=f>>>7,g=f^f<<24,f=h[u+1&7],g^=f^f>>>10,f=h[u+3&7],g^=f^f>>>3,f=h[u+4&7],g^=f^f<<7,f=h[u+7&7],f=f^f<<13,g^=f^f<<9,h[u]=g,c.i=u+1&7,g};function l(h,u){var f,g=[];if(u===(u|0))g[0]=u;else for(u=""+u,f=0;f<u.length;++f)g[f&7]=g[f&7]<<15^u.charCodeAt(f)+g[f+1&7]<<13;for(;g.length<8;)g.push(0);for(f=0;f<8&&g[f]===0;++f);for(f==8?g[7]=-1:g[f],h.x=g,h.i=0,f=256;f>0;--f)h.next()}l(c,i)}function o(i,c){return c.x=i.x.slice(),c.i=i.i,c}function a(i,c){i==null&&(i=+new Date);var l=new r(i),h=c&&c.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,g=(l.next()>>>0)/4294967296,w=(f+g)/(1<<21);while(w===0);return w},u.int32=l.next,u.quick=u,h&&(h.x&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=a:this.xorshift7=a})(Ot,e)})(Nn);var Uh=Nn.exports,Mn={exports:{}};Mn.exports;(function(e){(function(t,n,s){function r(i){var c=this;c.next=function(){var h=c.w,u=c.X,f=c.i,g,w;return c.w=h=h+1640531527|0,w=u[f+34&127],g=u[f=f+1&127],w^=w<<13,g^=g<<17,w^=w>>>15,g^=g>>>12,w=u[f]=w^g,c.i=f,w+(h^h>>>16)|0};function l(h,u){var f,g,w,y,$,E=[],_=128;for(u===(u|0)?(g=u,u=null):(u=u+"\0",g=0,_=Math.max(_,u.length)),w=0,y=-32;y<_;++y)u&&(g^=u.charCodeAt((y+32)%u.length)),y===0&&($=g),g^=g<<10,g^=g>>>15,g^=g<<4,g^=g>>>13,y>=0&&($=$+1640531527|0,f=E[y&127]^=g+$,w=f==0?w+1:0);for(w>=128&&(E[(u&&u.length||0)&127]=-1),w=127,y=4*128;y>0;--y)g=E[w+34&127],f=E[w=w+1&127],g^=g<<13,f^=f<<17,g^=g>>>15,f^=f>>>12,E[w]=g^f;h.w=$,h.X=E,h.i=w}l(c,i)}function o(i,c){return c.i=i.i,c.w=i.w,c.X=i.X.slice(),c}function a(i,c){i==null&&(i=+new Date);var l=new r(i),h=c&&c.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,g=(l.next()>>>0)/4294967296,w=(f+g)/(1<<21);while(w===0);return w},u.int32=l.next,u.quick=u,h&&(h.X&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=a:this.xor4096=a})(Ot,e)})(Mn);var Gh=Mn.exports,Cn={exports:{}};Cn.exports;(function(e){(function(t,n,s){function r(i){var c=this,l="";c.next=function(){var u=c.b,f=c.c,g=c.d,w=c.a;return u=u<<25^u>>>7^f,f=f-g|0,g=g<<24^g>>>8^w,w=w-u|0,c.b=u=u<<20^u>>>12^f,c.c=f=f-g|0,c.d=g<<16^f>>>16^w,c.a=w-u|0},c.a=0,c.b=0,c.c=-1640531527,c.d=1367130551,i===Math.floor(i)?(c.a=i/4294967296|0,c.b=i|0):l+=i;for(var h=0;h<l.length+20;h++)c.b^=l.charCodeAt(h)|0,c.next()}function o(i,c){return c.a=i.a,c.b=i.b,c.c=i.c,c.d=i.d,c}function a(i,c){var l=new r(i),h=c&&c.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,g=(l.next()>>>0)/4294967296,w=(f+g)/(1<<21);while(w===0);return w},u.int32=l.next,u.quick=u,h&&(typeof h=="object"&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=a:this.tychei=a})(Ot,e)})(Cn);var zh=Cn.exports,er={exports:{}};const qh={},Hh=Object.freeze(Object.defineProperty({__proto__:null,default:qh},Symbol.toStringTag,{value:"Module"})),Vh=wr(Hh);(function(e){(function(t,n,s){var r=256,o=6,a=52,i="random",c=s.pow(r,o),l=s.pow(2,a),h=l*2,u=r-1,f;function g(v,S,T){var N=[];S=S==!0?{entropy:!0}:S||{};var D=E($(S.entropy?[v,k(n)]:v??_(),3),N),A=new w(N),B=function(){for(var C=A.g(o),P=c,O=0;C<l;)C=(C+O)*r,P*=r,O=A.g(1);for(;C>=h;)C/=2,P/=2,O>>>=1;return(C+O)/P};return B.int32=function(){return A.g(4)|0},B.quick=function(){return A.g(4)/4294967296},B.double=B,E(k(A.S),n),(S.pass||T||function(C,P,O,K){return K&&(K.S&&y(K,A),C.state=function(){return y(A,{})}),O?(s[i]=C,P):C})(B,D,"global"in S?S.global:this==s,S.state)}function w(v){var S,T=v.length,N=this,D=0,A=N.i=N.j=0,B=N.S=[];for(T||(v=[T++]);D<r;)B[D]=D++;for(D=0;D<r;D++)B[D]=B[A=u&A+v[D%T]+(S=B[D])],B[A]=S;(N.g=function(C){for(var P,O=0,K=N.i,J=N.j,at=N.S;C--;)P=at[K=u&K+1],O=O*r+at[u&(at[K]=at[J=u&J+P])+(at[J]=P)];return N.i=K,N.j=J,O})(r)}function y(v,S){return S.i=v.i,S.j=v.j,S.S=v.S.slice(),S}function $(v,S){var T=[],N=typeof v,D;if(S&&N=="object")for(D in v)try{T.push($(v[D],S-1))}catch{}return T.length?T:N=="string"?v:v+"\0"}function E(v,S){for(var T=v+"",N,D=0;D<T.length;)S[u&D]=u&(N^=S[u&D]*19)+T.charCodeAt(D++);return k(S)}function _(){try{var v;return f&&(v=f.randomBytes)?v=v(r):(v=new Uint8Array(r),(t.crypto||t.msCrypto).getRandomValues(v)),k(v)}catch{var S=t.navigator,T=S&&S.plugins;return[+new Date,t,T,t.screen,k(n)]}}function k(v){return String.fromCharCode.apply(0,v)}if(E(s.random(),n),e.exports){e.exports=g;try{f=Vh}catch{}}else s["seed"+i]=g})(typeof self<"u"?self:Ot,[],Math)})(er);var jh=er.exports,Xh=Lh,Jh=Wh,Yh=Kh,Zh=Uh,Qh=Gh,tf=zh,Wt=jh;Wt.alea=Xh;Wt.xor128=Jh;Wt.xorwow=Yh;Wt.xorshift7=Zh;Wt.xor4096=Qh;Wt.tychei=tf;var Bn=Wt;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Fn{constructor(t,n,s,r,o){this.mean=t,this.stdDev=n,this.dtype=s,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const a=o||Math.random();this.random=Bn.alea(a.toString())}nextValue(){if(!isNaN(this.nextVal)){const r=this.nextVal;return this.nextVal=NaN,r}let t,n,s=!1;for(;!s;){let r,o,a;do r=2*this.random()-1,o=2*this.random()-1,a=r*r+o*o;while(a>=1||a===0);const i=Math.sqrt(-2*Math.log(a)/a);t=this.mean+this.stdDev*r*i,n=this.mean+this.stdDev*o*i,(!this.truncated||this.isValidTruncated(t))&&(s=!0)}return(!this.truncated||this.isValidTruncated(n))&&(this.nextVal=this.convertValue(n)),this.convertValue(t)}convertValue(t){return this.dtype==null||this.dtype==="float32"?t:Math.round(t)}isValidTruncated(t){return t<=this.upper&&t>=this.lower}}class ef{constructor(t,n,s,r){this.alpha=t,this.beta=1/n,this.dtype=s;const o=r||Math.random();this.randu=Bn.alea(o.toString()),this.randn=new Fn(0,1,s,!1,this.randu()),t<1?this.d=t+2/3:this.d=t-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let t,n,s,r,o,a;for(;;){do r=this.randn.nextValue(),a=1+this.c*r;while(a<=0);if(a*=a*a,t=r*r,n=1-.331*t*t,s=.5*t+this.d*(1-a+Math.log(a)),o=this.randu(),o<n||Math.log(o)<s)break}return a=1/this.beta*this.d*a,this.alpha<1&&(a*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(a)}convertValue(t){return this.dtype==="float32"?t:Math.round(t)}}class nf{constructor(t=0,n=1,s,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=t,this.range=n-t,this.dtype=s,r==null&&(r=Math.random()),typeof r=="number"&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${t} - ${n} <= 1 and dtype is not float`);this.random=Bn.alea(r)}convertValue(t){return this.canReturnFloat()?t:Math.round(t)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sf(e,t,n=1,s="float32",r){if(ot(e),n==null&&(n=1),s==null&&(s="float32"),s!=="float32"&&s!=="int32")throw new Error(`Unsupported data type ${s}`);const o=new ef(t,n,s,r),a=Et(e,s);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const Gb=m({randomGamma_:sf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rf(e,t=0,n=1,s,r){if(ot(e),s!=null&&s==="bool")throw new Error(`Unsupported data type ${s}`);const o=new Fn(t,n,s,!1,r),a=Et(e,s);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const of=m({randomNormal_:rf});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function af(e,t,n){if(t!=null&&t==="bool")throw new Error(`Unsupported data type ${t}`);return of(e,0,1,t,n)}const zb=m({randomStandardNormal_:af});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cf(e,t=0,n=1,s="float32",r){ot(e);const o=Et(e,s),a=new nf(t,n,null,r);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const nr=m({randomUniform_:cf});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lf(e,t,n,s){return nr(e,t,n,"int32",s)}const qb=m({randomUniformInt_:lf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ne(e,t,n=1,s="float32"){if(n===0)throw new Error("Cannot have a step of zero");const r={start:e,stop:t,step:n,dtype:s};return b.runKernel(Ma,{},r)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uf(e){const n={input:d(e,"input","real")};return b.runKernel(Ca,n)}const Me=m({real_:uf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hf(e){const n={x:d(e,"x","reciprocal")};return b.runKernel(Ba,n)}const Hb=m({reciprocal_:hf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ff(e){const n={x:d(e,"x","relu")};return b.runKernel(Fa,n)}const Rn=m({relu_:ff});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function df(e){const n={x:d(e,"x","relu6")};return b.runKernel(La,n)}const pf=m({relu6_:df});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gf(e,t){const s={x:d(e,"x","reverse")},r={dims:t};return b.runKernel(Wa,s,r)}const te=m({reverse_:gf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mf(e){const t=d(e,"x","reverse");return p(t.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),te(t,0)}const Vb=m({reverse1d_:mf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bf(e,t){const n=d(e,"x","reverse");return p(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),te(n,t)}const jb=m({reverse2d_:bf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wf(e,t){const n=d(e,"x","reverse");return p(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),te(n,t)}const Xb=m({reverse3d_:wf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yf(e,t){const n=d(e,"x","reverse");return p(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),te(n,t)}const Jb=m({reverse4d_:yf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $f(e){const n={x:d(e,"x","round")};return b.runKernel(Ka,n)}const kf=m({round_:$f});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xf(e){const n={x:d(e,"x","rsqrt","float32")};return b.runKernel(Ua,n)}const Yb=m({rsqrt_:xf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ef(e){const n={x:d(e,"x","selu")};return b.runKernel(Va,n)}const Zb=m({selu_:Ef});function vf(e,t,n,s,r,o=[1,1],a="NHWC"){const i=d(e,"x","separableConv2d"),c=d(t,"depthwiseFilter","separableConv2d"),l=d(n,"pointwiseFilter","separableConv2d");let h=i,u=!1;if(i.rank===3&&(u=!0,h=x(i,[1,i.shape[0],i.shape[1],i.shape[2]])),a==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");p(h.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${h.rank}.`),p(c.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${c.rank}.`),p(l.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${c.rank}.`),p(l.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${l.shape[0]}.`),p(l.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${l.shape[1]}.`);const f=c.shape[2],g=c.shape[3];p(l.shape[2]===f*g,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${f*g}, but got ${l.shape[2]}.`);const w=Js(h,c,s,r,a,o),$=vn(w,l,1,"valid",a);return u?x($,[$.shape[1],$.shape[2],$.shape[3]]):$}const Qb=m({separableConv2d_:vf});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Sf(e,t){const n=d(e,"x","setdiff1d"),s=d(t,"y","setdiff1d");p(n.dtype===s.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${s.dtype}).`),p(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),p(s.rank===1,()=>`y should be 1D tensor, but got y (${s.shape}).`);const r=await n.data(),o=await s.data(),a=new Set(o);let i=0;for(let h=0;h<r.length;h++)a.has(r[h])||i++;const c=new Ze([i],n.dtype),l=new Ze([i],"int32");for(let h=0,u=0;h<r.length;h++)a.has(r[h])||(c.values[u]=r[h],l.values[u]=h,u++);return[c.toTensor(),l.toTensor()]}const tw=Sf;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function If(e){const n={x:d(e,"x","sign")};return b.runKernel(Ya,n)}const ew=m({sign_:If});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tf(e){const n={x:d(e,"x","sin","float32")};return b.runKernel(Xa,n)}const nw=m({sin_:Tf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Df(e){const n={x:d(e,"x","sinh")};return b.runKernel(Ja,n)}const sw=m({sinh_:Df});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Af(e,t,n){const s=d(e,"x","slice1d");return p(s.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${s.rank} tensor`),q(s,[t],[n])}const rw=m({slice1d_:Af});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _f(e,t,n){const s=d(e,"x","slice2d");return p(s.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${s.rank} tensor`),q(s,t,n)}const ow=m({slice2d_:_f});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nf(e,t,n){const s=d(e,"x","slice3d");return p(s.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${s.rank} tensor`),q(s,t,n)}const aw=m({slice3d_:Nf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mf(e,t,n){const s=d(e,"x","slice4d");return p(s.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${s.rank} tensor`),q(s,t,n)}const iw=m({slice4d_:Mf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cf(e,t=-1){const n=d(e,"logits","softmax","float32");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);const s={logits:n},r={dim:t};return b.runKernel(ri,s,r)}const cw=m({softmax_:Cf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bf(e){p(e.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);const t={input:e};return b.runKernel(Bo,t)}const sr=m({fft_:Bf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ff(e){p(e.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);const t={input:e};return b.runKernel(zo,t)}const fn=m({ifft_:Ff});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rf(e){const t=e.shape[e.shape.length-1],n=e.size/t;let s;if(t<=2){const r=x(e,[n,t]);s=fn(r)}else{const r=[n,2*(t-1)],o=x(Me(e),[n,t]),a=x(Tn(e),[n,t]),i=te(q(o,[0,1],[n,t-2]),1),c=I(te(q(a,[0,1],[n,t-2]),1),W(-1)),l=rt([o,i],1),h=rt([a,c],1),u=x(Ct(l,h),[r[0],r[1]]);s=fn(u)}if(s=Me(s),e.rank===3&&e.shape[0]!==0){const r=s,o=e.shape[0];s=x(s,[o,s.shape[0]/o,s.shape[1]]),r.dispose()}return s}const Pf=m({irfft_:Rf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Of(e,t,n=0){const r={x:d(e,"x","split")},o={numOrSizeSplits:t,axis:n};return b.runKernel(si,r,o)}const Ce=m({split_:Of});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lf(e,t){p(e.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1];const s=e.size/n;let r;if(t!=null&&t<n){const w=e.shape.map($=>0),y=e.shape.map($=>$);y[e.shape.length-1]=t,r=q(e,w,y),n=t}else if(t!=null&&t>n){const w=e.shape.map(y=>y);w[e.shape.length-1]=t-n,r=rt([e,pe(w)],e.shape.length-1),n=t}else r=e;const o=ct(r),a=x(Ct(r,o),[s,n]),i=sr(a),c=Math.floor(n/2)+1,l=Me(i),h=Tn(i),u=Ce(l,[c,n-c],l.shape.length-1),f=Ce(h,[c,n-c],h.shape.length-1),g=r.shape.slice();return g[r.shape.length-1]=c,x(Ct(u[0],f[0]),g)}const rr=m({rfft_:Lf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wf(e,t){let n=d(e,"a","squaredDifference"),s=d(t,"b","squaredDifference");[n,s]=V(n,s),Y(n.shape,s.shape);const r={a:n,b:s},o={};return b.runKernel(ui,r,o)}const Kf=m({squaredDifference_:Wf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uf(e,t){const n=d(e,"x","squeeze","string_or_numeric");return x(n,Er(n.shape,t).newShape)}const or=m({squeeze_:Uf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gf(e,t=0){const n=Te(e,"tensors","stack","string_or_numeric");p(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&p(t<=n[0].rank,()=>"Axis must be <= rank of the tensor");const s=n,r={axis:t};return b.runKernel(va,s,r)}const Be=m({stack_:Gf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zf(e,t=0){const s={x:d(e,"x","step")},r={alpha:t};return b.runKernel(Si,s,r)}const qf=m({step_:zf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hf(e,t,n,s,r=0,o=0,a=0,i=0,c=0){const h={x:d(e,"x","stridedSlice","string_or_numeric")},u={begin:t,end:n,strides:s,beginMask:r,endMask:o,ellipsisMask:a,newAxisMask:i,shrinkAxisMask:c};return b.runKernel(fi,h,u)}const lw=m({stridedSlice_:Hf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vf(e){const n={x:d(e,"x","tan","float32")};return b.runKernel(bi,n)}const uw=m({tan_:Vf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ft(e,t){ee(e);const n=St(e,t);if(n.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return It(e,null,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xe(e,t,n){if(ee(e),t!=null&&t.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const s=St(e,n);if(s.length!==2&&s.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return It(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jf(e,t,n){if(ee(e),t!=null&&t.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const s=St(e,n);if(s.length!==3&&s.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return It(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hw(e,t,n){if(ee(e),t!=null&&t.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const s=St(e,n);if(s.length!==4&&s.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return It(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fw(e,t,n){if(ee(e),t!=null&&t.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const s=St(e,n);if(s.length!==5&&s.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return It(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dw(e,t,n){if(ee(e),t!=null&&t.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const s=St(e,n);if(s.length!==6&&s.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return t=t||s,It(e,t,s,n)}function Xf(e,t,n){const s=t.rank>1?t.shape[t.rank-1]:1,r=t.rank>1?t.rank-1:1,o=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${s}, and batchDim: ${r}.`;if(n.rank<r)throw new Error(o+` update.rank < ${r}. `);if(e.length<s+(n.rank-r))throw new Error(o+` Output shape length < ${s+(n.rank-r)}`);if(n.rank!==r+e.length-s)throw new Error(o+` update.rank != ${r+e.length-s}`);for(let a=0;a<r;++a)if(n.shape[a]!==t.shape[a])throw new Error(o+` updates.shape[${a}] (${n.shape[a]}) != indices.shape[${a}] (${t.shape[a]}).`);for(let a=0;a<n.rank-r;++a)if(n.shape[a+r]!==e[a+s])throw new Error(o+` updates.shape[${a+r}] (${n.shape[a+r]}) != shape[${a+r}] (${e[a+r]})`)}function ar(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Xf(n,t,e)}function pw(e,t,n){const s=t.shape.length,r=s>1?t.shape[s-1]:1,o=n.length;let a=1;for(let u=r;u<o;++u)a*=n[u];const i=r<1?1:r,c=H(t.shape)/i,l=[...ge(n.slice(0,r)),1],h=H(n);return{sliceRank:r,numUpdates:c,sliceSize:a,strides:l,outputSize:h}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jf(e,t,n){const s=d(e,"tensor","tensorScatterupdate"),r=d(t,"indices","tensorScatterupdate","int32"),o=d(n,"updates","tensorScatterupdate");if(ar(o,r,s.shape),s.dtype!==o.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${s.dtype} and ${o.dtype}.`);const a={tensor:s,indices:r,updates:o},i={};return b.runKernel(za,a,i)}const gw=m({tensorScatterUpdate_:Jf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yf(e,t=1,n=!0){const s=d(e,"x","topk");if(s.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const r=s.shape[s.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>r)throw new Error(`'k' passed to topk() must be <= the last dimension (${r}) but got ${t}`);const o={x:s},a={k:t,sorted:n},[i,c]=b.runKernel(yi,o,a);return{values:i,indices:c}}const mw=m({topk_:Yf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zf(e,t=0,n=1,s,r){if(ot(e),s!=null&&s==="bool")throw new Error("Unsupported data type $ { dtype }");const o=new Fn(t,n,s,!0,r),a=Et(e,s);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const bw=m({truncatedNormal_:Zf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qf(e,t=0){const n=d(e,"x","unique","string_or_numeric");p(n.rank>0,()=>"The input tensor must be at least 1D");const s={x:n},r={axis:t},[o,a]=b.runKernel(ki,s,r);return{values:o,indices:a}}const ww=m({unique_:Qf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function td(e,t,n){const s=d(e,"x","unsortedSegmentSum"),r=d(t,"segmentIds","unsortedSegmentSum","int32");p(ue(n),()=>"numSegments must be of dtype int");const o={x:s,segmentIds:r},a={numSegments:n};return b.runKernel(Ei,o,a)}const yw=m({unsortedSegmentSum_:td});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ed(e,t=0){const n=d(e,"x","unstack","string_or_numeric");p(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);const s={value:n},r={axis:t};return b.runKernel(xi,s,r)}const ir=m({unstack_:ed});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $w(e,t){return tr(e,t,"right")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kw(e,t=!0,n,s){return b.makeVariable(e,t,n,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nd(e,t){const n=[];for(let o=0;o<t.length;o++)t[o]&&n.push(o);const s=Et(e,"int32"),r=Et([n.length,e.length],"int32");for(let o=0;o<n.length;o++){const a=s.indexToLoc(n[o]),i=o*e.length;r.values.set(a,i)}return r.toTensor()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function sd(e){const t=d(e,"condition","whereAsync","bool"),n=await t.data(),s=nd(t.shape,n);return e!==t&&t.dispose(),s}const rd=sd;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function od(e,t,n){const s=d(e,"tensor","boolMask"),r=d(t,"mask","boolMask","bool"),o=n??0,a=r.rank,i=s.shape;p(a>0,()=>"mask cannot be scalar"),ut(i.slice(o,o+a),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let c=1;for(let y=o;y<o+a;y++)c*=i[y];const l=i.slice(0,o).concat([c],i.slice(o+a)),h=x(s,l),u=x(r,[-1]),f=await rd(u),g=or(f,[1]),w=Iu(h,g,o);return e!==s&&s.dispose(),t!==r&&r.dispose(),g.dispose(),h.dispose(),u.dispose(),f.dispose(),w}const xw=od;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ad(e,t,n){const s=d(e,"x","transpose");if(t==null&&(t=s.shape.map((a,i)=>i).reverse()),p(s.rank===t.length,()=>`Error in transpose: rank of input ${s.rank} must match length of perm ${t}.`),t.forEach(a=>{p(a>=0&&a<s.rank,()=>`All entries in 'perm' must be between 0 and ${s.rank-1} but got ${t}`)}),s.rank<=1)return s.clone();const r={x:s},o={perm:t};return s.dtype==="complex64"?X(()=>{let a=Me(s),i=Tn(s);return a=b.runKernel(Ue,{x:a},o),i=b.runKernel(Ue,{x:i},o),n&&(i=wt(i)),Ct(a,i)}):b.runKernel(Ue,r,o)}const cs=m({transpose_:ad});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function id(e,t,n,s,r=!0){const o=d(e,"v","movingAverage"),a=d(t,"x","movingAverage"),i=d(n,"decay","movingAverage");Li(o,a),p(Lt(o.shape,a.shape),()=>"Shape mismatch in v and x");const c=W(1),l=F(c,i);let h=I(F(a,o),l);if(r){p(s!=null,()=>"When using zeroDebias: true, step is required.");const u=d(s,"step","movingAverage");h=G(h,F(c,Ae(i,u)))}return M(o,h)}const Ew=m({movingAverage_:id});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cd(e,t,n){ot(n);const s=d(e,"indices","scatterND","int32"),r=d(t,"updates","scatterND");ar(r,s,n);const o={indices:s,updates:r},a={shape:n};return b.runKernel(Ga,o,a)}const vw=m({scatterND_:cd});function ld(e,t,n,s){if(e.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);const r=e.rank>0?e.shape[0]:1,o=e.rank>1?e.shape[1]:1;if(n.length!==o)throw new Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${o}.`);const a=t.size;if(!(t.rank===0||t.rank===1&&a===r))throw new Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${r}]`);if(t.dtype!==s.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ud(e,t,n,s=0){ot(n);const r=d(e,"sparseIndices","sparseToDense","int32"),o=d(t,"sparseValues","sparseToDense","string_or_numeric"),a=d(s,"defaultValue","sparseToDense",o.dtype);ld(r,o,n,a);const i={sparseIndices:r,sparseValues:o,defaultValue:a},c={outputShape:n};return b.runKernel(li,i,c)}const Sw=m({sparseToDense_:ud});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hd(e,t){const n=d(t,"indices","gatherND","int32"),r={params:d(e,"x","gatherND","string_or_numeric"),indices:n};return b.runKernel(Ko,r)}const Iw=m({gatherND_:hd});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fd(e,t){if(t==null)return e.shape.slice();if(Lt(e.shape,t))return t;if(e.shape.length===t.length){const n=[];for(let s=0;s<e.shape.length;s++)t[s]==null&&e.shape[s]!=null?n.push(e.shape[s]):n.push(t[s]);return n}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dd(e,t,n,s){const r=d(e,"x","dropout");if(p(r.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${r.dtype} tensor instead.`),p(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof Q?r.clone():r;const o=fd(r,n),a=1-t,i=G(vu(M(nr(o,0,1,"float32",s),a)),a);return I(r,i)}const Tw=m({dropout_:dd});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pd(e){return Math.floor(Math.pow(2,Math.ceil(Math.log(e)/Math.log(2))))}function cr(e,t,n){const s=1-e%2,r=new Float32Array(e);for(let o=0;o<e;++o){const a=2*Math.PI*o/(e+s-1);r[o]=t-n*Math.cos(a)}return ft(r,"float32")}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function gd(e,t,n=1){const s=d(e,"predictions","inTopK"),r=d(t,"targets","inTopK");p(s.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${s.rank}`),p(s.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${s.rank} and targets rank ${r.rank}`),ut(s.shape.slice(0,s.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const o=s.shape[s.shape.length-1];p(n>0&&n<=o,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${o}), but got ${n}`);const a=await s.data(),i=await r.data(),[c,l]=[a.length/o,o],h=vr("bool",c);for(let u=0;u<c;u++){const f=u*l,g=a.subarray(f,f+l),w=[];for(let y=0;y<g.length;y++)w.push({value:g[y],index:y});w.sort((y,$)=>$.value-y.value),h[u]=0;for(let y=0;y<n;y++)if(w[y].index===i[u]){h[u]=1;break}}return e!==s&&s.dispose(),t!==r&&r.dispose(),ie(h,r.shape,"bool")}const Dw=gd;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function md(e,t,n,s,r,o="NHWC",a){let i=e;e.rank===3&&(i=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]])),p(i.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${i.shape}.`),p(c.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${c.shape}.`),p(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const l=o==="NHWC"?i.shape[3]:i.shape[1],h=o==="NHWC"?c.shape[3]:c.shape[1];p(l===n[2],()=>`Error in conv2dDerFilter: depth of input ${l}) must match input depth in filter (${n[2]}.`),p(h===n[3],()=>`Error in conv2dDerFilter: depth of dy (${h}) must match output depth for filter (${n[3]}).`),pt("conv2dDerFilter",r,a);const u={x:i,dy:c},f={strides:s,pad:r,dataFormat:o,dimRoundingMode:a,filterShape:n};return b.runKernel(co,u,f)}const bd=m({conv2DBackpropFilter_:md});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pn(e,t,n){if(n==null||n==="linear")return e;if(n==="relu")return I(e,qf(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function On(e,t){let n=t;const s=Jl(e.shape,t.shape);return s.length>0&&(n=U(n,s)),x(n,e.shape)}function Ln(e,t,n,s){if(t==="linear")return e;if(t==="relu")return Rn(e);if(t==="elu")return ou(e);if(t==="relu6")return pf(e);if(t==="prelu")return Ch(e,n);if(t==="leakyrelu")return Fu(e,s);if(t==="sigmoid")return ce(e);throw new Error(`Unknown fused activation ${t}.`)}const Wn=(e,t)=>!(e>0)||t==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wd({x:e,filter:t,strides:n,pad:s,dataFormat:r="NHWC",dilations:o=[1,1],dimRoundingMode:a,bias:i,activation:c="linear",preluActivationWeights:l,leakyreluAlpha:h}){if(c=c||"linear",Wn(b.state.gradientDepth,c)===!1){p(r==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${r} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let T=vn(e,t,n,s,r,o,a);return i!=null&&(T=M(T,i)),Ln(T,c,l,h)}const u=d(e,"x","conv2d","float32"),f=d(t,"filter","conv2d","float32");let g=u,w=!1;u.rank===3&&(w=!0,g=x(u,[1,u.shape[0],u.shape[1],u.shape[2]])),p(g.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${g.rank}.`),p(f.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${f.rank}.`),pt("fused conv2d",s,a);const y=r==="NHWC"?g.shape[3]:g.shape[1];p(f.shape[2]===y,()=>`Error in conv2d: depth of input (${y}) must match input depth for filter ${f.shape[2]}.`),p(Tt(n,o),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`);const $=Pe(g.shape,f.shape,n,o,s,a);let E;i!=null&&(E=d(i,"bias","fused conv2d"),[E]=V(E,u),r==="NHWC"?Y($.outShape,E.shape):(p(E.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${E.shape.length}.`),p(E.shape.length===0||E.shape[0]===$.outChannels||E.shape[0]===1,()=>`Error in fused conv2d: bias shape (${E.shape}) is not compatible with the number of output channels (${$.outChannels})`)));let _;if(l!=null){const T=l.shape;if(p(T.length<=1||T.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${T.length}.`),T.length===1)p(T[0]===1||T[0]===$.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${T}) is not compatible with the number of output channels (${$.outChannels}).`);else if(T.length===3)try{Y(T,$.outShape)}catch{const D=`Error in fused conv2d: PReLU activation weights (${T}) is not compatible with the output shape of the conv2d (${$.outShape}).`;throw Error(D)}_=d(l,"prelu weights","fused conv2d")}const k=(T,N)=>{p(r==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${r} but only NHWC is currently supported.`);const[D,A,B,C]=N,P=Pn(T,B,c);p(De(o),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${o}'`);const O=Xs(A.shape,P,D,n,s),K=bd(A,P,D.shape,n,s),J=[O,K];if(C!=null){const at=On(C,P);J.push(at)}return J},v={x:g,filter:f,bias:E,preluActivationWeights:_},S={strides:n,pad:s,dataFormat:r,dilations:o,dimRoundingMode:a,activation:c,leakyreluAlpha:h};return i==null?yt((N,D,A)=>{let B=b.runKernel(jn,v,S);return A([D,N,B]),w&&(B=x(B,[B.shape[1],B.shape[2],B.shape[3]])),{value:B,gradFunc:k}})(g,f):yt((N,D,A,B)=>{let C=b.runKernel(jn,v,S);return B([D,N,C,A]),w&&(C=x(C,[C.shape[1],C.shape[2],C.shape[3]])),{value:C,gradFunc:k}})(g,f,E)}const yd=m({fusedConv2d_:wd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $d(e,t,n,s,r,o=[1,1],a){let i=e;e.rank===3&&(i=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const l={x:i,dy:c},h={strides:s,pad:r,dimRoundingMode:a,dilations:o,filterShape:n};return b.runKernel(ko,l,h)}const kd=m({depthwiseConv2dNativeBackpropFilter_:$d});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xd(e,t,n,s,r,o=[1,1],a){let i=t,c=!1;t.rank===3&&(c=!0,i=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const l={dy:i,filter:n},h={strides:s,pad:r,dimRoundingMode:a,dilations:o,inputShape:e},u=b.runKernel(xo,l,h);return c?x(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const Ed=m({depthwiseConv2dNativeBackpropInput_:xd});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vd({x:e,filter:t,strides:n,pad:s,dataFormat:r="NHWC",dilations:o=[1,1],dimRoundingMode:a,bias:i,activation:c="linear",preluActivationWeights:l,leakyreluAlpha:h}){if(Wn(b.state.gradientDepth,c)===!1){let S=Js(e,t,n,s,r,o,a);return i!=null&&(S=M(S,i)),Ln(S,c,l,h)}const u=d(e,"x","depthwiseConv2d","float32"),f=d(t,"filter","depthwiseConv2d","float32");let g=u,w=!1;u.rank===3&&(w=!0,g=x(u,[1,u.shape[0],u.shape[1],u.shape[2]])),p(g.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${g.rank}.`),p(f.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${f.rank}.`),p(g.shape[3]===f.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${g.shape[3]}) must match the inChannels dimension in filter ${f.shape[2]}.`),o==null&&(o=[1,1]),p(Tt(n,o),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),pt("fused depthwiseConv2d",s,a);const y=Pe(g.shape,f.shape,n,o,s,a,!0);let $;i!=null&&($=d(i,"bias","fused conv2d"),[$]=V($,u),Y(y.outShape,$.shape));let E;l!=null&&(E=d(l,"prelu weights","fused depthwiseConv2d"));const _=(S,T)=>{p(De(o),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${o}'`);const[N,D,A,B]=T,C=Pn(S,A,c),P=Ed(D.shape,C,N,n,s,o,a),O=kd(D,C,N.shape,n,s,o,a);if(B!=null){const K=On($,C);return[P,O,K]}return[P,O]},k={x:g,filter:f,bias:$,preluActivationWeights:E},v={strides:n,pad:s,dataFormat:r,dilations:o,dimRoundingMode:a,activation:c,leakyreluAlpha:h};return i==null?yt((T,N,D)=>{let A=b.runKernel(Xn,k,v);return D([N,T,A]),w&&(A=x(A,[A.shape[1],A.shape[2],A.shape[3]])),{value:A,gradFunc:_}})(g,f):yt((T,N,D,A)=>{let B=b.runKernel(Xn,k,v);return A([N,T,B,D]),w&&(B=x(B,[B.shape[1],B.shape[2],B.shape[3]])),{value:B,gradFunc:_}})(g,f,$)}const Sd=m({fusedDepthwiseConv2d_:vd});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Id({a:e,b:t,transposeA:n=!1,transposeB:s=!1,bias:r,activation:o="linear",preluActivationWeights:a,leakyreluAlpha:i=.2}){if(Wn(b.state.gradientDepth,o)===!1){let C=L(e,t,n,s);return r!=null&&(C=M(C,r)),Ln(C,o,a,i)}let c=d(e,"a","fused matMul"),l=d(t,"b","fused matMul");[c,l]=V(c,l);const h=n?c.shape[c.rank-2]:c.shape[c.rank-1],u=s?l.shape[l.rank-1]:l.shape[l.rank-2],f=n?c.shape[c.rank-1]:c.shape[c.rank-2],g=s?l.shape[l.rank-2]:l.shape[l.rank-1],w=c.shape.slice(0,-2),y=l.shape.slice(0,-2),$=H(w),E=H(y);p(h===u,()=>`Error in fused matMul: inner shapes (${h}) and (${u}) of Tensors with shapes ${c.shape} and ${l.shape} and transposeA=${n} and transposeB=${s} must match.`);const k=Y(c.shape.slice(0,-2),l.shape.slice(0,-2)).concat([f,g]),v=n?x(c,[$,h,f]):x(c,[$,f,h]),S=s?x(l,[E,g,u]):x(l,[E,u,g]);let T;r!=null&&(T=d(r,"bias","fused matMul"),[T]=V(T,c),Y(k,T.shape));let N;a!=null&&(N=d(a,"prelu weights","fused matMul"));const D=(C,P)=>{const[O,K,J,at]=P,mt=Pn(x(C,J.shape),J,o);let Ut,Gt;if(!n&&!s?(Ut=L(mt,K,!1,!0),Gt=L(O,mt,!0,!1)):!n&&s?(Ut=L(mt,K,!1,!1),Gt=L(mt,O,!0,!1)):n&&!s?(Ut=L(K,mt,!1,!0),Gt=L(O,mt,!1,!1)):(Ut=L(K,mt,!0,!0),Gt=L(mt,O,!0,!0)),r!=null){const br=On(at,mt);return[Ut,Gt,br]}else return[Ut,Gt]},A={a:v,b:S,bias:T,preluActivationWeights:N},B={transposeA:n,transposeB:s,activation:o,leakyreluAlpha:i};return r==null?yt((P,O,K)=>{const J=b.runKernel(Vn,A,B);return K([P,O,J]),{value:x(J,k),gradFunc:D}})(v,S):yt((P,O,K,J)=>{const at=b.runKernel(Vn,A,B);return J([P,O,at,K]),{value:x(at,k),gradFunc:D}})(v,S,T)}const Td=m({fusedMatMul_:Id});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Aw=Object.freeze(Object.defineProperty({__proto__:null,conv2d:yd,depthwiseConv2d:Sd,matMul:Td},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dd(e){return cr(e,.54,.46)}const Ad=m({hammingWindow_:Dd});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _d(e){return cr(e,.5,.5)}const lr=m({hannWindow_:_d});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nd(e,t,n,s=!1,r=0){let o=0;const a=[];for(;o+t<=e.size;)a.push(q(e,o,t)),o+=n;if(s)for(;o<e.size;){const i=o+t-e.size,c=rt([q(e,o,t-i),Oe([i],r)]);a.push(c),o+=n}return a.length===0?xe([],[0,t]):x(rt(a),[a.length,t])}const ur=m({frame_:Nd});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Md(e,t,n,s,r=lr){s==null&&(s=pd(t));const o=ur(e,t,n),a=I(o,r(t));return rr(a,s)}const Cd=m({stft_:Md});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bd(e,t,n,s,r="bilinear",o=0){const a=d(e,"image","cropAndResize"),i=d(t,"boxes","cropAndResize","float32"),c=d(n,"boxInd","cropAndResize","int32"),l=i.shape[0];p(a.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${a.rank}.`),p(i.rank===2&&i.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${l},4] but had shape ${i.shape}.`),p(c.rank===1&&c.shape[0]===l,()=>`Error in cropAndResize: boxInd must be have size [${l}] but had shape ${i.shape}.`),p(s.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${s.length}.`),p(s[0]>=1&&s[1]>=1,()=>`cropSize must be atleast [1,1], but was ${s}`),p(r==="bilinear"||r==="nearest",()=>`method must be bilinear or nearest, but was ${r}`);const h={image:a,boxes:i,boxInd:c},u={method:r,extrapolationValue:o,cropSize:s};return b.runKernel(bo,h,u)}const Fd=m({cropAndResize_:Bd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rd(e){const t=d(e,"image","flipLeftRight","float32");p(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const n={image:t};return b.runKernel(Ro,n,{})}const Pd=m({flipLeftRight_:Rd});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Od(e){const t=d(e,"image","grayscaleToRGB"),n=t.rank-1,s=t.shape[n];p(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),p(s===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${s}.`);const r=new Array(t.rank);return r.fill(1,0,n),r[n]=3,ke(t,r)}const Ld=m({grayscaleToRGB_:Od});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wd(e){const t=d(e,"image","RGBToGrayscale"),n=t.rank-1,s=t.shape[n];p(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),p(s===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${s}.`);const r=t.dtype,o=j(t,"float32"),a=ft([.2989,.587,.114]);let i;switch(t.rank){case 2:i=re("ij,j->i",o,a);break;case 3:i=re("ijk,k->ij",o,a);break;case 4:i=re("ijkl,l->ijk",o,a);break;case 5:i=re("ijklm,m->ijkl",o,a);break;case 6:i=re("ijklmn,n->ijklm",o,a);break;default:throw new Error("Not a valid tensor rank.")}return i=At(i,-1),j(i,r)}const Kd=m({rgbToGrayscale_:Wd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ud(e,t,n=0,s=.5){const r=d(e,"image","rotateWithOffset","float32");p(r.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${r.rank}.`);const o={image:r},a={radians:t,fillValue:n,center:s};return b.runKernel(Ii,o,a)}const Gd=m({rotateWithOffset_:Ud});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ne(e,t,n,s,r,o){s==null&&(s=.5),r==null&&(r=Number.NEGATIVE_INFINITY),o==null&&(o=0);const a=e.shape[0];return n=Math.min(n,a),p(0<=s&&s<=1,()=>`iouThreshold must be in [0, 1], but was '${s}'`),p(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),p(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),p(t.rank===1,()=>"scores must be a 1D tensor"),p(t.shape[0]===a,()=>`scores has incompatible shape with boxes. Expected ${a}, but was ${t.shape[0]}`),p(0<=o&&o<=1,()=>`softNmsSigma must be in [0, 1], but was '${o}'`),{maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:o}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zd(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY){const o=d(e,"boxes","nonMaxSuppression","float32"),a=d(t,"scores","nonMaxSuppression","float32"),i=ne(o,a,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const c={maxOutputSize:n,iouThreshold:s,scoreThreshold:r};return b.runKernel(ya,{boxes:o,scores:a},c)}const qd=m({nonMaxSuppression_:zd});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hd(e,t,n){const s=Vd(e,t,n),r=s<0?-(s+1):s;e.splice(r,0,t)}function Vd(e,t,n){return Xd(e,t,n||jd)}function jd(e,t){return e>t?1:e<t?-1:0}function Xd(e,t,n){let s=0,r=e.length,o=0,a=!1;for(;s<r;){o=s+(r-s>>>1);const i=n(t,e[o]);i>0?s=o+1:(r=o,a=!i)}return a?s:-s-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jd(e,t,n,s,r){return Kn(e,t,n,s,r,0)}function Yd(e,t,n,s,r,o){return Kn(e,t,n,s,r,0,!1,o,!0)}function Zd(e,t,n,s,r,o){return Kn(e,t,n,s,r,o,!0)}function Kn(e,t,n,s,r,o,a=!1,i=!1,c=!1){const l=[];for(let $=0;$<t.length;$++)t[$]>r&&l.push({score:t[$],boxIndex:$,suppressBeginIndex:0});l.sort(ls);const h=o>0?-.5/o:0,u=[],f=[];for(;u.length<n&&l.length>0;){const $=l.pop(),{score:E,boxIndex:_,suppressBeginIndex:k}=$;if(E<r)break;let v=!1;for(let S=u.length-1;S>=k;--S){const T=Qd(e,_,u[S]);if(T>=s){v=!0;break}if($.score=$.score*tp(s,h,T),$.score<=r)break}$.suppressBeginIndex=u.length,v||($.score===E?(u.push(_),f.push($.score)):$.score>r&&Hd(l,$,ls))}const g=u.length,w=n-g;i&&w>0&&(u.push(...new Array(w).fill(0)),f.push(...new Array(w).fill(0)));const y={selectedIndices:u};return a&&(y.selectedScores=f),c&&(y.validOutputs=g),y}function Qd(e,t,n){const s=e.subarray(t*4,t*4+4),r=e.subarray(n*4,n*4+4),o=Math.min(s[0],s[2]),a=Math.min(s[1],s[3]),i=Math.max(s[0],s[2]),c=Math.max(s[1],s[3]),l=Math.min(r[0],r[2]),h=Math.min(r[1],r[3]),u=Math.max(r[0],r[2]),f=Math.max(r[1],r[3]),g=(i-o)*(c-a),w=(u-l)*(f-h);if(g<=0||w<=0)return 0;const y=Math.max(o,l),$=Math.max(a,h),E=Math.min(i,u),_=Math.min(c,f),k=Math.max(E-y,0)*Math.max(_-$,0);return k/(g+w-k)}function tp(e,t,n){const s=Math.exp(t*n*n);return n<=e?s:0}function ls(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function ep(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY){const o=d(e,"boxes","nonMaxSuppressionAsync"),a=d(t,"scores","nonMaxSuppressionAsync"),i=ne(o,a,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const c=await Promise.all([o.data(),a.data()]),l=c[0],h=c[1],{selectedIndices:u}=Jd(l,h,n,s,r);return o!==e&&o.dispose(),a!==t&&a.dispose(),ft(u,"int32")}const np=ep;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sp(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,o=0){const a=d(e,"boxes","nonMaxSuppression"),i=d(t,"scores","nonMaxSuppression"),c=ne(a,i,n,s,r,o);n=c.maxOutputSize,s=c.iouThreshold,r=c.scoreThreshold,o=c.softNmsSigma;const l={boxes:a,scores:i},h={maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:o},u=b.runKernel(ka,l,h);return{selectedIndices:u[0],selectedScores:u[1]}}const rp=m({nonMaxSuppressionWithScore_:sp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function op(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,o=0){const a=d(e,"boxes","nonMaxSuppressionAsync"),i=d(t,"scores","nonMaxSuppressionAsync"),c=ne(a,i,n,s,r,o);n=c.maxOutputSize,s=c.iouThreshold,r=c.scoreThreshold,o=c.softNmsSigma;const l=await Promise.all([a.data(),i.data()]),h=l[0],u=l[1],{selectedIndices:f,selectedScores:g}=Zd(h,u,n,s,r,o);return a!==e&&a.dispose(),i!==t&&i.dispose(),{selectedIndices:ft(f,"int32"),selectedScores:ft(g)}}const ap=op;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ip(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,o=!1){const a=d(e,"boxes","nonMaxSuppression"),i=d(t,"scores","nonMaxSuppression"),c=ne(a,i,n,s,r,null),l=c.maxOutputSize,h=c.iouThreshold,u=c.scoreThreshold,f={boxes:a,scores:i},g={maxOutputSize:l,iouThreshold:h,scoreThreshold:u,padToMaxOutputSize:o},w=b.runKernel($a,f,g);return{selectedIndices:w[0],validOutputs:w[1]}}const cp=m({nonMaxSuppressionPadded_:ip});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function lp(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,o=!1){const a=d(e,"boxes","nonMaxSuppressionAsync"),i=d(t,"scores","nonMaxSuppressionAsync"),c=ne(a,i,n,s,r,null),l=c.maxOutputSize,h=c.iouThreshold,u=c.scoreThreshold,[f,g]=await Promise.all([a.data(),i.data()]),{selectedIndices:w,validOutputs:y}=Yd(f,g,l,h,u,o);return a!==e&&a.dispose(),i!==t&&i.dispose(),{selectedIndices:ft(w,"int32"),validOutputs:W(y,"int32")}}const up=lp;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hp(e,t,n=!1,s=!1){const r=d(e,"images","resizeBilinear");p(r.rank===3||r.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),p(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),p(s===!1||n===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let o=r,a=!1;r.rank===3&&(a=!0,o=x(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:o},c={alignCorners:n,halfPixelCenters:s,size:t},l=b.runKernel(Oa,i,c);return a?x(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const fp=m({resizeBilinear_:hp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dp(e,t,n=!1,s=!1){const r=d(e,"images","resizeNearestNeighbor");p(r.rank===3||r.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),p(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),p(r.dtype==="float32"||r.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),p(s===!1||n===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let o=r,a=!1;r.rank===3&&(a=!0,o=x(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:o},c={alignCorners:n,halfPixelCenters:s,size:t},l=b.runKernel(Pa,i,c);return a?x(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const pp=m({resizeNearestNeighbor_:dp});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gp(e,t="binary",n=!1,s=.5){const r=d(e,"image","threshold"),o=.2989,a=.587,i=.114,c=r.shape[0]*r.shape[1];let l=I(ft([s]),255),h,u,f,g;if(p(r.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${r.rank}.`),p(r.shape[2]===3||r.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${r.shape[2]}.`),p(r.dtype==="int32"||r.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${r.dtype}.`),p(t==="otsu"||t==="binary",()=>`Method must be binary or otsu, but was ${t}`),r.shape[2]===3){[h,u,f]=Ce(r,[1,1,1],-1);const $=I(h,o),E=I(u,a),_=I(f,i);g=M(M($,E),_)}else g=e;if(t==="otsu"){const $=El(j(kf(g),"int32"),ie([]),256);l=mp($,c)}const w=n?Qs(g,l):In(g,l);return j(I(w,255),"int32")}function mp(e,t){let n=ft([-1]),s=ft([0]),r=ft([0]),o,a,i,c,l,h;for(let u=0;u<e.size-1;u++){o=q(e,0,u+1),a=q(e,u+1),l=G(U(o),t),h=G(U(a),t);const f=U(I(o,Ne(0,o.size)));i=G(f,U(o));const g=Oe(a.shape,o.size),w=M(Ne(0,a.size),g),y=I(a,w);c=G(U(y),U(a));const $=F(i,c),E=F(i,c),_=I(l,h);r=I(I(_,$),E);const k=In(r,s);s=Mt(k,r,s),n=Mt(k,ft([u]),n)}return n}const bp=m({threshold_:gp});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wp(e,t,n="nearest",s="constant",r=0,o){const a=d(e,"image","transform","float32"),i=d(t,"transforms","transform","float32");p(a.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${a.rank}.`),p(i.rank===2&&(i.shape[0]===a.shape[0]||i.shape[0]===1)&&i.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),p(o==null||o.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${o}.`);const c={image:a,transforms:i},l={interpolation:n,fillMode:s,fillValue:r,outputShape:o};return b.runKernel($i,c,l)}const yp=m({transform_:wp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $p(e,t,n){const s=d(e,"a","bandPart");p(s.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${s.rank}.`);const r=s.shape,[o,a]=s.shape.slice(-2);let i,c;typeof t=="number"?(p(t%1===0,()=>`bandPart(): numLower must be an integer, got ${t}.`),p(t<=o,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${o}).`),i=d(t<0?o:t,"numLower","bandPart")):(p(t.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),i=Mt(is(t,0),o,hn(t,o))),typeof n=="number"?(p(n%1===0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),p(n<=a,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${a}).`),c=d(n<0?a:n,"numUpper","bandPart")):(p(n.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),c=Mt(is(n,0),a,hn(n,a)));const l=x(Ne(0,o,1,"int32"),[-1,1]),h=Ne(0,a,1,"int32"),u=F(l,h),f=ln(Qs(u,i),Au(u,wt(c))),g=pe([o,a],s.dtype);return x(Be(ir(x(s,[-1,o,a])).map(w=>Mt(f,w,g))),r)}const kp=m({bandPart_:$p});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xp(e){let t;if(Array.isArray(e)){t=!1,p(e!=null&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const r=e[0].shape[0];for(let o=1;o<e.length;++o)p(e[o].shape[0]===r,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[o].shape[0]} vs. ${r})`)}else t=!0,e=Ce(e,e.shape[0],0).map(r=>or(r,[0]));p(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);const n=[],s=e;for(let r=0;r<e.length;++r)n.push(b.tidy(()=>{let o=s[r];if(r>0)for(let a=0;a<r;++a){const i=I(U(I(n[a],o)),n[a]);o=F(o,i)}return G(o,Sn(o,"euclidean"))}));return t?Be(n,0):n}const Ep=m({gramSchmidt_:xp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vp(e,t=!1){if(p(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return us(e,t);{const n=e.shape.slice(0,e.shape.length-2).reduce((c,l)=>c*l),s=ir(x(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),r=[],o=[];s.forEach(c=>{const[l,h]=us(c,t);r.push(l),o.push(h)});const a=x(Be(r,0),e.shape),i=x(Be(o,0),e.shape);return[a,i]}}function us(e,t=!1){return b.tidy(()=>{p(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);const n=e.shape[0],s=e.shape[1];let r=xu(n),o=Xt(e);const a=xe([[1]],[1,1]);let i=Xt(a);const c=n>=s?s:n;for(let l=0;l<c;++l){const h=o,u=i,f=r;[i,o,r]=b.tidy(()=>{const g=q(o,[l,l],[n-l,1]),w=Sn(g),y=q(o,[l,l],[1,1]),$=Mt(In(y,0),xe([[-1]]),xe([[1]])),E=F(y,I($,w)),_=G(g,E);_.shape[0]===1?i=Xt(a):i=rt([a,q(_,[1,0],[_.shape[0]-1,_.shape[1]])],0);const k=wt(G(L($,E),w)),v=q(o,[l,0],[n-l,s]),S=I(k,i),T=cs(i);if(l===0)o=F(v,L(S,L(T,v)));else{const A=F(v,L(S,L(T,v)));o=rt([q(o,[0,0],[l,s]),A],0)}const N=cs(S),D=q(r,[0,l],[n,r.shape[1]-l]);if(l===0)r=F(D,L(L(D,i),N));else{const A=F(D,L(L(D,i),N));r=rt([q(r,[0,0],[n,l]),A],1)}return[i,o,r]}),st([h,u,f])}return!t&&n>s&&(r=q(r,[0,0],[n,s]),o=q(o,[0,0],[s,s])),[r,o]})}const Sp=m({qr_:vp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var tt;(function(e){e[e.NONE=0]="NONE",e[e.MEAN=1]="MEAN",e[e.SUM=2]="SUM",e[e.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"})(tt||(tt={}));function Ip(e,t,n=tt.SUM_BY_NONZERO_WEIGHTS){const s=d(e,"losses","computeWeightedLoss");let r=null;t!=null&&(r=d(t,"weights","computeWeightedLoss"));const o=r==null?s:I(s,r);if(n===tt.NONE)return o;if(n===tt.SUM)return U(o);if(n===tt.MEAN){if(r==null)return un(o);{const a=s.size/r.size,i=G(U(o),U(r));return a>1?G(i,W(a)):i}}if(n===tt.SUM_BY_NONZERO_WEIGHTS){if(r==null)return G(U(o),W(s.size));{const a=I(r,Vt(s.shape)),i=j(U(wh(a,W(0))),"float32");return G(U(o),i)}}throw Error(`Unknown reduction: ${n}`)}const $t=m({computeWeightedLoss_:Ip});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tp(e,t,n,s=tt.SUM_BY_NONZERO_WEIGHTS){const r=d(e,"labels","absoluteDifference"),o=d(t,"predictions","absoluteDifference");let a=null;n!=null&&(a=d(n,"weights","absoluteDifference")),ut(r.shape,o.shape,"Error in absoluteDifference: ");const i=it(F(r,o));return $t(i,a,s)}const Dp=m({absoluteDifference_:Tp});function Ap(e,t,n,s,r=tt.SUM_BY_NONZERO_WEIGHTS){const o=d(e,"labels","cosineDistance"),a=d(t,"predictions","cosineDistance");let i=null;s!=null&&(i=d(s,"weights","cosineDistance")),ut(o.shape,a.shape,"Error in cosineDistance: ");const c=W(1),l=F(c,U(I(o,a),n,!0));return $t(l,i,r)}const _p=m({cosineDistance_:Ap});function Np(e,t,n,s=tt.SUM_BY_NONZERO_WEIGHTS){let r=d(e,"labels","hingeLoss");const o=d(t,"predictions","hingeLoss");let a=null;n!=null&&(a=d(n,"weights","hingeLoss")),ut(r.shape,o.shape,"Error in hingeLoss: ");const i=W(1);r=F(I(W(2),r),i);const c=Rn(F(i,I(r,o)));return $t(c,a,s)}const Mp=m({hingeLoss_:Np});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cp(e,t,n,s=1,r=tt.SUM_BY_NONZERO_WEIGHTS){const o=d(e,"labels","huberLoss"),a=d(t,"predictions","huberLoss");let i=null;n!=null&&(i=d(n,"weights","huberLoss")),ut(o.shape,a.shape,"Error in huberLoss: ");const c=W(s),l=it(F(a,o)),h=hn(l,c),u=F(l,h),f=M(I(W(.5),gt(h)),I(c,u));return $t(f,i,r)}const Bp=m({huberLoss_:Cp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fp(e,t,n,s=1e-7,r=tt.SUM_BY_NONZERO_WEIGHTS){const o=d(e,"labels","logLoss"),a=d(t,"predictions","logLoss");let i=null;n!=null&&(i=d(n,"weights","logLoss")),ut(o.shape,a.shape,"Error in logLoss: ");const c=W(1),l=W(s),h=wt(I(o,_e(M(a,l)))),u=I(F(c,o),_e(M(F(c,a),l))),f=F(h,u);return $t(f,i,r)}const Rp=m({logLoss_:Fp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pp(e,t,n,s=tt.SUM_BY_NONZERO_WEIGHTS){const r=d(e,"labels","meanSquaredError"),o=d(t,"predictions","meanSquaredError");let a=null;n!=null&&(a=d(n,"weights","meanSquaredError")),ut(r.shape,o.shape,"Error in meanSquaredError: ");const i=Kf(r,o);return $t(i,a,s)}const Op=m({meanSquaredError_:Pp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lp(e,t){const n=d(e,"labels","sigmoidCrossEntropyWithLogits"),s=d(t,"logits","sigmoidCrossEntropyWithLogits");ut(n.shape,s.shape,"Error in sigmoidCrossEntropyWithLogits: ");const r=Rn(s),o=I(s,n),a=Ku(Qt(wt(it(s))));return M(F(r,o),a)}function Wp(e,t,n,s=0,r=tt.SUM_BY_NONZERO_WEIGHTS){let o=d(e,"multiClassLabels","sigmoidCrossEntropy");const a=d(t,"logits","sigmoidCrossEntropy");let i=null;if(n!=null&&(i=d(n,"weights","sigmoidCrossEntropy")),ut(o.shape,a.shape,"Error in sigmoidCrossEntropy: "),s>0){const l=W(s),h=W(1),u=W(.5);o=M(I(o,F(h,l)),I(u,l))}const c=Lp(o,a);return $t(c,i,r)}const Kp=m({sigmoidCrossEntropy_:Wp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Up(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return yt((r,o,a)=>{const c=Ju(o,[n],!0),l=F(j(o,"float32"),c);a([r,l]);const h=wt(I(l,r));return{value:U(h,[n]),gradFunc:(g,w)=>{const[y,$]=w,E=Le(g.shape,[n]);return[I(x(g,E),F(j(y,"float32"),Qt($))),I(x(g,E),F(Qt($),j(y,"float32")))]}}})(e,t)}function Gp(e,t,n,s=0,r=tt.SUM_BY_NONZERO_WEIGHTS){let o=d(e,"onehotLabels","softmaxCrossEntropy");const a=d(t,"logits","softmaxCrossEntropy");let i=null;if(n!=null&&(i=d(n,"weights","softmaxCrossEntropy")),ut(o.shape,a.shape,"Error in softmaxCrossEntropy: "),s>0){const l=W(s),h=W(1),u=W(o.shape[1]);o=M(I(o,F(h,l)),G(l,u))}const c=Up(o,a);return $t(c,i,r)}const zp=m({softmaxCrossEntropy_:Gp});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qp(e,t,n,s){const r=d(e,"indices","sparseFillEmptyRows","int32"),o=d(t,"values","sparseFillEmptyRows"),a=d(n,"denseShape","sparseFillEmptyRows","int32"),i=d(s,"defaultValue","sparseFillEmptyRows",o.dtype);if(r.rank!==2)throw new Error(`Indices should be Tensor2D but received shape
        ${r.shape}`);if(o.rank!==1)throw new Error(`Values should be Tensor1D but received shape ${o.shape}`);if(a.rank!==1)throw new Error(`Dense shape should be Tensor1D but received shape ${a.shape}`);if(i.rank!==0)throw new Error(`Default value should be a scalar but received shape ${i.shape}`);const c={indices:r,values:o,denseShape:a,defaultValue:i},l=b.runKernel(oi,c);return{outputIndices:l[0],outputValues:l[1],emptyRowIndicator:l[2],reverseIndexMap:l[3]}}const Hp=m({sparseFillEmptyRows_:qp});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vp(e,t,n){const s=d(e,"inputIndices","sparseReshape","int32"),r=d(t,"inputShape","sparseReshape","int32"),o=d(n,"newShape","sparseReshape","int32");if(s.rank!==2)throw new Error(`Input indices should be Tensor2D but received shape
        ${s.shape}`);if(r.rank!==1)throw new Error(`Input shape should be Tensor1D but received shape ${r.shape}`);if(o.rank!==1)throw new Error(`New shape should be Tensor1D but received shape ${o.shape}`);const a={inputIndices:s,inputShape:r,newShape:o},i=b.runKernel(ai,a);return{outputIndices:i[0],outputShape:i[1]}}const jp=m({sparseReshape_:Vp});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xp(e,t,n){const s=d(e,"data","sparseSegmentMean"),r=d(t,"indices","sparseSegmentMean","int32"),o=d(n,"segmentIds","sparseSegmentMean","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
          ${r.shape}`);if(o.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
          ${o.shape}`);const a={data:s,indices:r,segmentIds:o};return b.runKernel(ii,a)}const Jp=m({sparseSegmentMean_:Xp});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yp(e,t,n){const s=d(e,"data","sparseSegmentSum"),r=d(t,"indices","sparseSegmentSum","int32"),o=d(n,"segmentIds","sparseSegmentSum","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
         ${r.shape}`);if(o.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
         ${o.shape}`);const a={data:s,indices:r,segmentIds:o};return b.runKernel(ci,a)}const Zp=m({sparseSegmentSum_:Yp});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qp(e,t,n,s,r,o,a,i){const c=d(e,"data","stringNGrams","string");if(c.dtype!=="string")throw new Error("Data must be of datatype string");if(c.shape.length!==1)throw new Error(`Data must be a vector, saw: ${c.shape}`);const l=d(t,"dataSplits","stringNGrams");if(l.dtype!=="int32")throw new Error("Data splits must be of datatype int32");const h={separator:n,nGramWidths:s,leftPad:r,rightPad:o,padWidth:a,preserveShortSequences:i},u={data:c,dataSplits:l},f=b.runKernel(di,u,h);return{nGrams:f[0],nGramsSplits:f[1]}}const tg=m({stringNGrams_:Qp});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eg(e,t,n=!0){const s=d(e,"input","stringSplit","string"),r=d(t,"delimiter","stringSplit","string");if(s.rank!==1)throw new Error(`Input should be Tensor1D but received shape ${s.shape}`);if(r.rank!==0)throw new Error(`Delimiter should be a scalar but received shape ${r.shape}`);const o={skipEmpty:n},a={input:s,delimiter:r},i=b.runKernel(pi,a,o);return{indices:i[0],values:i[1],shape:i[2]}}const ng=m({stringSplit_:eg});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sg(e,t){const n=d(e,"input","stringToHashBucketFast","string"),s={numBuckets:t};if(t<=0)throw new Error("Number of buckets must be at least 1");const r={input:n};return b.runKernel(gi,r,s)}const rg=m({stringToHashBucketFast_:sg});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function og(e,t,n,s=!0){const r=d(e,"input","staticRegexReplace","string"),o={pattern:t,rewrite:n,replaceGlobal:s};return b.runKernel(hi,{x:r},o)}const ag=m({staticRegexReplace_:og});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _w={fft:sr,ifft:fn,rfft:rr,irfft:Pf},Nw={hammingWindow:Ad,hannWindow:lr,frame:ur,stft:Cd},Mw={flipLeftRight:Pd,grayscaleToRGB:Ld,resizeNearestNeighbor:pp,resizeBilinear:fp,rgbToGrayscale:Kd,rotateWithOffset:Gd,cropAndResize:Fd,nonMaxSuppression:qd,nonMaxSuppressionAsync:np,nonMaxSuppressionWithScore:rp,nonMaxSuppressionWithScoreAsync:ap,nonMaxSuppressionPadded:cp,nonMaxSuppressionPaddedAsync:up,threshold:bp,transform:yp},Cw={bandPart:kp,gramSchmidt:Ep,qr:Sp},Bw={absoluteDifference:Dp,computeWeightedLoss:$t,cosineDistance:_p,hingeLoss:Mp,huberLoss:Bp,logLoss:Rp,meanSquaredError:Op,sigmoidCrossEntropy:Kp,softmaxCrossEntropy:zp},Fw={sparseFillEmptyRows:Hp,sparseReshape:jp,sparseSegmentMean:Jp,sparseSegmentSum:Zp},Rw={stringNGrams:tg,stringSplit:ng,stringToHashBucketFast:rg,staticRegexReplace:ag};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ig=new Map,cg=new Map;class lg{getClassName(){return this.constructor.className}static fromConfig(t,n){return new t(n)}}class _t{constructor(){this.classNameMap={}}static getMap(){return _t.instance==null&&(_t.instance=new _t),_t.instance}static register(t){_t.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function ug(e,t,n){p(e.className!=null,()=>"Class being registered does not have the static className property defined."),p(typeof e.className=="string",()=>"className is required to be a string, but got type "+typeof e.className),p(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof t>"u"&&(t="Custom"),typeof n>"u"&&(n=e.className);const s=n,r=t+">"+s;return _t.register(e),ig.set(r,e),cg.set(e,r),e}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Kt extends lg{minimize(t,n=!1,s){const{value:r,grads:o}=this.computeGradients(t,s);if(s!=null){const a=s.map(i=>({name:i.name,tensor:o[i.name]}));this.applyGradients(a)}else this.applyGradients(o);return st(o),n?r:(r.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,n){return Uu(t,n)}dispose(){this.iterations_!=null&&st(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:W(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(Kt,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hg extends Kt{static get className(){return"Adadelta"}constructor(t,n,s=null){super(),this.learningRate=t,this.rho=n,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=b.backend.epsilon())}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=b.registeredVariables[s],a=!1;this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accum_grad`,variable:X(()=>ct(o).variable(a))}),this.accumulatedUpdates[r]==null&&(this.accumulatedUpdates[r]={originalName:`${s}/accum_var`,variable:X(()=>ct(o).variable(a))});const i=Array.isArray(t)?t[r].tensor:t[s];if(i==null)return;const c=this.accumulatedGrads[r].variable,l=this.accumulatedUpdates[r].variable;X(()=>{const h=M(I(c,this.rho),I(gt(i),1-this.rho)),u=I(G(vt(M(l,this.epsilon)),vt(M(c,this.epsilon))),i),f=M(I(l,this.rho),I(gt(u),1-this.rho));c.assign(h),l.assign(f);const g=M(I(u,-this.learningRate),o);o.assign(g)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(st(this.accumulatedGrads.map(t=>t.variable)),st(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=t.length/2,s=!1;this.accumulatedGrads=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedUpdates=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.rho,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fg extends Kt{static get className(){return"Adagrad"}constructor(t,n=.1){super(),this.learningRate=t,this.initialAccumulatorValue=n,this.accumulatedGrads=[]}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=b.registeredVariables[s];this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accumulator`,variable:X(()=>Oe(o.shape,this.initialAccumulatorValue).variable(!1))});const a=Array.isArray(t)?t[r].tensor:t[s];if(a==null)return;const i=this.accumulatedGrads[r].variable;X(()=>{const c=M(i,gt(a));i.assign(c);const l=M(I(G(a,vt(M(c,b.backend.epsilon()))),-this.learningRate),o);o.assign(l)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&st(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulatedGrads=t.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,n){return new t(n.learningRate,n.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class dg extends Kt{static get className(){return"Adam"}constructor(t,n,s,r=null){super(),this.learningRate=t,this.beta1=n,this.beta2=s,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],X(()=>{this.accBeta1=W(n).variable(),this.accBeta2=W(s).variable()}),r==null&&(this.epsilon=b.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);X(()=>{const s=F(1,this.accBeta1),r=F(1,this.accBeta2);n.forEach((o,a)=>{const i=b.registeredVariables[o],c=!1;this.accumulatedFirstMoment[a]==null&&(this.accumulatedFirstMoment[a]={originalName:`${o}/m`,variable:X(()=>ct(i).variable(c))}),this.accumulatedSecondMoment[a]==null&&(this.accumulatedSecondMoment[a]={originalName:`${o}/v`,variable:X(()=>ct(i).variable(c))});const l=Array.isArray(t)?t[a].tensor:t[o];if(l==null)return;const h=this.accumulatedFirstMoment[a].variable,u=this.accumulatedSecondMoment[a].variable,f=M(I(h,this.beta1),I(l,1-this.beta1)),g=M(I(u,this.beta2),I(gt(l),1-this.beta2)),w=G(f,s),y=G(g,r);h.assign(f),u.assign(g);const $=M(I(G(w,M(vt(y),this.epsilon)),-this.learningRate),i);i.assign($)}),this.accBeta1.assign(I(this.accBeta1,this.beta1)),this.accBeta2.assign(I(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&st(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&st(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t),X(()=>{this.accBeta1.assign(Ae(this.beta1,this.iterations_+1)),this.accBeta2.assign(Ae(this.beta2,this.iterations_+1))});const n=t.length/2,s=!1;this.accumulatedFirstMoment=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedSecondMoment=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class pg extends Kt{static get className(){return"Adamax"}constructor(t,n,s,r=null,o=0){super(),this.learningRate=t,this.beta1=n,this.beta2=s,this.epsilon=r,this.decay=o,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],X(()=>{this.iteration=W(0).variable(),this.accBeta1=W(n).variable()}),r==null&&(this.epsilon=b.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);X(()=>{const s=F(1,this.accBeta1),r=G(-this.learningRate,M(I(this.iteration,this.decay),1));n.forEach((o,a)=>{const i=b.registeredVariables[o],c=!1;this.accumulatedFirstMoment[a]==null&&(this.accumulatedFirstMoment[a]={originalName:`${o}/m`,variable:ct(i).variable(c)}),this.accumulatedWeightedInfNorm[a]==null&&(this.accumulatedWeightedInfNorm[a]={originalName:`${o}/v`,variable:ct(i).variable(c)});const l=Array.isArray(t)?t[a].tensor:t[o];if(l==null)return;const h=this.accumulatedFirstMoment[a].variable,u=this.accumulatedWeightedInfNorm[a].variable,f=M(I(h,this.beta1),I(l,1-this.beta1)),g=I(u,this.beta2),w=it(l),y=lh(g,w);h.assign(f),u.assign(y);const $=M(I(G(r,s),G(f,M(y,this.epsilon))),i);i.assign($)}),this.iteration.assign(M(this.iteration,1)),this.accBeta1.assign(I(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&st(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&st(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon,n.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hr extends Kt{static get className(){return"SGD"}constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=Array.isArray(t)?t[r].tensor:t[s];if(o==null)return;const a=b.registeredVariables[s];X(()=>{const i=M(I(this.c,o),a);a.assign(i)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=Vi(W(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,n){return new t(n.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class gg extends hr{static get className(){return"Momentum"}constructor(t,n,s=!1){super(t),this.learningRate=t,this.momentum=n,this.useNesterov=s,this.accumulations=[],this.m=W(this.momentum)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=b.registeredVariables[s];this.accumulations[r]==null&&(this.accumulations[r]={originalName:`${s}/momentum`,variable:X(()=>ct(o).variable(!1))});const a=this.accumulations[r].variable,i=Array.isArray(t)?t[r].tensor:t[s];i!=null&&X(()=>{let c;const l=M(I(this.m,a),i);this.useNesterov?c=M(I(this.c,M(i,I(l,this.m))),o):c=M(I(this.c,l),o),a.assign(l),o.assign(c)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&st(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulations=t.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,n){return new t(n.learningRate,n.momentum,n.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class mg extends Kt{static get className(){return"RMSProp"}constructor(t,n=.9,s=0,r=null,o=!1){if(super(),this.learningRate=t,this.decay=n,this.momentum=s,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=o,r==null&&(this.epsilon=b.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=b.registeredVariables[s],a=!1;this.accumulatedMeanSquares[r]==null&&(this.accumulatedMeanSquares[r]={originalName:`${s}/rms`,variable:X(()=>ct(o).variable(a))}),this.accumulatedMoments[r]==null&&(this.accumulatedMoments[r]={originalName:`${s}/momentum`,variable:X(()=>ct(o).variable(a))}),this.accumulatedMeanGrads[r]==null&&this.centered&&(this.accumulatedMeanGrads[r]={originalName:`${s}/mg`,variable:X(()=>ct(o).variable(a))});const i=Array.isArray(t)?t[r].tensor:t[s];if(i==null)return;const c=this.accumulatedMeanSquares[r].variable,l=this.accumulatedMoments[r].variable;X(()=>{const h=M(I(c,this.decay),I(gt(i),1-this.decay));if(this.centered){const u=this.accumulatedMeanGrads[r].variable,f=M(I(u,this.decay),I(i,1-this.decay)),g=G(I(i,this.learningRate),vt(F(h,M(gt(f),this.epsilon)))),w=M(I(l,this.momentum),g);c.assign(h),u.assign(f),l.assign(w);const y=F(o,w);o.assign(y)}else{const u=M(I(c,this.decay),I(gt(i),1-this.decay)),f=M(I(l,this.momentum),G(I(i,this.learningRate),vt(M(u,this.epsilon))));c.assign(u),l.assign(f);const g=F(o,f);o.assign(g)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&st(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&st(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&st(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=this.centered?t.length/3:t.length/2,s=!1;this.accumulatedMeanSquares=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedMoments=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=t.slice(n*2,n*3).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,n){return new t(n.learningRate,n.decay,n.momentum,n.epsilon,n.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bg=[hg,fg,dg,pg,gg,mg,hr];function wg(){for(const e of bg)ug(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yg="model",$g=".json",kg=".weights.bin";function hs(e){return new Promise(t=>setTimeout(t)).then(e)}class Pt{constructor(t){if(!R().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");t.startsWith(Pt.URL_SCHEME)&&(t=t.slice(Pt.URL_SCHEME.length)),(t==null||t.length===0)&&(t=yg),this.modelJsonFileName=t+$g,this.weightDataFileName=t+kg}async save(t){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const n=dt.join(t.weightData),s=window.URL.createObjectURL(new Blob([n],{type:"application/octet-stream"}));if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:t.weightSpecs}],o=Ls(t,r),a=window.URL.createObjectURL(new Blob([JSON.stringify(o)],{type:"application/json"})),i=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=a,await hs(()=>i.dispatchEvent(new MouseEvent("click"))),t.weightData!=null){const c=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;c.download=this.weightDataFileName,c.href=s,await hs(()=>c.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:me(t)}}}}Pt.URL_SCHEME="downloads://";class xg{constructor(t){if(t==null||t.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${t}`);this.jsonFile=t[0],this.weightsFiles=t.slice(1)}async load(){return new Promise((t,n)=>{const s=new FileReader;s.onload=r=>{const o=JSON.parse(r.target.result),a=o.modelTopology;if(a==null){n(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(o.weightsManifest==null){n(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){t({modelTopology:a});return}const c=xn(o,l=>this.loadWeights(l));t(c)},s.onerror=r=>n(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(t){const n=[],s=[];for(const a of t)n.push(...a.weights),s.push(...a.paths);const r=this.checkManifestAndWeightFiles(t),o=s.map(a=>this.loadWeightsFile(a,r[a]));return Promise.all(o).then(a=>[n,a])}loadWeightsFile(t,n){return new Promise((s,r)=>{const o=new FileReader;o.onload=a=>{const i=a.target.result;s(i)},o.onerror=a=>r(`Failed to weights data from file of path '${t}'.`),o.readAsArrayBuffer(n)})}checkManifestAndWeightFiles(t){const n=[],s=this.weightsFiles.map(o=>rs(o.name)),r={};for(const o of t)o.paths.forEach(a=>{const i=rs(a);if(n.indexOf(i)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${i}'`);if(n.push(i),s.indexOf(i)===-1)throw new Error(`Weight file with basename '${i}' is not provided.`);r[a]=this.weightsFiles[s.indexOf(i)]});if(n.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${n.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const Eg=e=>R().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Pt.URL_SCHEME)?vg(e.slice(Pt.URL_SCHEME.length)):null;z.registerSaveRouter(Eg);function vg(e="model"){return new Pt(e)}function Sg(e){return new xg(e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fs(e,t,n,s){a(e),n=n??0,s=s??1,i(n,s);let r=0;const o=c=>(c.then(l=>{const h=n+ ++r/e.length*(s-n);return t(h),l}),c);function a(c){p(c!=null&&Array.isArray(c)&&c.length>0,()=>"promises must be a none empty array")}function i(c,l){p(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${c}`),p(l>=0&&l<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${l}`),p(l>=c,()=>`startFraction must be no more than endFraction, but got startFraction ${c} and endFraction ${l}`)}return Promise.all(e.map(o))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function fr(e,t){t==null&&(t={});const n=t.fetchFunc==null?R().platform.fetch:t.fetchFunc,s=e.map(u=>n(u,t.requestInit,{isBinary:!0})),i=(t.onProgress==null?await Promise.all(s):await fs(s,t.onProgress,0,.5)).map(u=>u.arrayBuffer());return t.onProgress==null?await Promise.all(i):await fs(i,t.onProgress,.5,1)}function Ig(e,t){var n;const s=t.fetchFunc==null?R().platform.fetch:t.fetchFunc;let r=0,o;return(n=t.onProgress)===null||n===void 0||n.call(t,0),new ReadableStream({pull:async a=>{for(var i;r<e.length;){o||(o=(await s(e[r],t.requestInit,{isBinary:!0})).body.getReader());const{done:c,value:l}=await o.read();if(c){r++,o=void 0,(i=t.onProgress)===null||i===void 0||i.call(t,r/e.length);continue}a.enqueue(l);return}a.close()}})}async function Tg(e,t="",n,s){return dr(a=>fr(a,{requestInit:s}))(e,t,n)}function dr(e){return async(t,n="",s)=>{const r=t.map(()=>!1),o={},a=s!=null?s.map(()=>!1):[],i=[];if(t.forEach((g,w)=>{let y=0;g.weights.forEach($=>{const E="quantization"in $?$.quantization.dtype:$.dtype,_=Bt[E]*H($.shape),k=()=>{r[w]=!0,o[w]==null&&(o[w]=[]),o[w].push({manifestEntry:$,groupOffset:y,sizeBytes:_})};s!=null?s.forEach((v,S)=>{v===$.name&&(k(),a[S]=!0)}):k(),i.push($.name),y+=_})}),!a.every(g=>g)){const g=s.filter((w,y)=>!a[y]);throw new Error(`Could not find weights in manifest with names: ${g.join(", ")}. 
Manifest JSON has weights with names: ${i.join(", ")}.`)}const c=r.reduce((g,w,y)=>(w&&g.push(y),g),[]),l=[];c.forEach(g=>{t[g].paths.forEach(w=>{const y=n+(n.endsWith("/")?"":"/")+w;l.push(y)})});const h=await e(l),u={};let f=0;return c.forEach(g=>{const w=t[g].paths.length,y=new dt(h.slice(f,f+w));o[g].forEach(E=>{const _=y.slice(E.groupOffset,E.groupOffset+E.sizeBytes),k=Ps(_,[E.manifestEntry]);for(const v in k)u[v]=k[v]}),f+=w}),u}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dg="application/octet-stream",Ag="application/json";class Un{constructor(t,n){if(this.DEFAULT_METHOD="POST",n==null&&(n={}),this.weightPathPrefix=n.weightPathPrefix,this.weightUrlConverter=n.weightUrlConverter,n.fetchFunc!=null?(p(typeof n.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=n.fetchFunc):this.fetch=R().platform.fetch,p(t!=null&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&p(t.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,n.requestInit!=null&&n.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=n.requestInit||{},this.loadOptions=n}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const n=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);n.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],r=Ls(t,s);if(n.body.append("model.json",new Blob([JSON.stringify(r)],{type:Ag}),"model.json"),t.weightData!=null){const a=dt.join(t.weightData);n.body.append("model.weights.bin",new Blob([a],{type:Dg}),"model.weights.bin")}const o=await this.fetch(this.path,n);if(o.ok)return{modelArtifactsInfo:me(t),responses:[o]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${o.status}.`)}async loadModelJSON(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code ${t.status}. Please verify this URL points to the model JSON of the model to load.`);let n;try{n=await t.json()}catch{let a=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?a+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":a+=" Please make sure the server is serving valid JSON for this request.",new Error(a)}const s=n.modelTopology,r=n.weightsManifest;if(s==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return n}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const t=await this.loadModelJSON();return xn(t,n=>this.loadWeights(n))}async loadStream(){const t=await this.loadModelJSON(),n=await this.getWeightUrls(t.weightsManifest),s=sn(t.weightsManifest),r=()=>Ig(n,this.loadOptions);return Object.assign(Object.assign({},t),{weightSpecs:s,getWeightStream:r})}async getWeightUrls(t){const n=Array.isArray(this.path)?this.path[1]:this.path,[s,r]=_g(n),o=this.weightPathPrefix||s,a=[],i=[];for(const c of t)for(const l of c.paths)this.weightUrlConverter!=null?i.push(this.weightUrlConverter(l)):a.push(o+l+r);return this.weightUrlConverter&&a.push(...await Promise.all(i)),a}async loadWeights(t){const n=await this.getWeightUrls(t),s=sn(t),r=await fr(n,this.loadOptions);return[s,r]}}Un.URL_SCHEME_REGEX=/^https?:\/\//;function _g(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),s=e.substring(0,t),r=n>t?e.substring(n):"";return[s+"/",r]}function dn(e){return e.match(Un.URL_SCHEME_REGEX)!=null}const pr=(e,t)=>{if(typeof fetch>"u"&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(Array.isArray(e)?n=e.every(s=>dn(s)):n=dn(e),n)return Gn(e,t)}return null};z.registerSaveRouter(pr);z.registerLoadRouter(pr);function Gn(e,t){return new Un(e,t)}function Ng(e,t){return Gn(e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ve{constructor(t){this.modelArtifacts=t}load(){return this.modelArtifacts}}class gr{constructor(t){this.saveHandler=t}save(t){return this.saveHandler(t)}}class Mg{constructor(t){t.load&&(this.load=()=>Promise.resolve(t.load())),t.save&&(this.save=n=>Promise.resolve(t.save(n)))}}function Cg(e,t,n,s){const r=arguments;return new Mg(mr(...r))}function mr(e,t,n,s){return arguments.length===1?e.modelTopology!=null||e.weightSpecs!=null?new Ve(e):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Ve({modelTopology:e})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Ve({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:s}))}function Bg(e){return new gr(e)}function Fg(e){return new gr(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pw=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:dt,browserFiles:Sg,browserHTTPRequest:Ng,concatenateArrayBuffers:sc,copyModel:Sc,decodeWeights:Ps,decodeWeightsStream:Qi,encodeWeights:Ji,fromMemory:Cg,fromMemorySync:mr,getLoadHandlers:hc,getModelArtifactsForJSON:xn,getModelArtifactsForJSONSync:Ws,getModelArtifactsInfoForJSON:me,getSaveHandlers:uc,getWeightSpecs:sn,http:Gn,isHTTPScheme:dn,listModels:Ec,loadWeights:Tg,moveModel:Ic,registerLoadRouter:lc,registerSaveRouter:cc,removeModel:vc,weightsLoaderFactory:dr,withSaveHandler:Bg,withSaveHandlerSync:Fg},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Dt,ds=!1;function Rg(e,t=3){if(t>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(e==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,s=!1,r=!1,o=!1,a=!1,i=!1;if(e.data instanceof Uint8Array)n=!0;else if(typeof ImageData<"u"&&e instanceof ImageData)s=!0;else if(typeof HTMLVideoElement<"u"&&e instanceof HTMLVideoElement)r=!0;else if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement)o=!0;else if(e.getContext!=null)a=!0;else if(typeof ImageBitmap<"u"&&e instanceof ImageBitmap)i=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);if(ve(Hn,b.backendName)!=null){const w={pixels:e},y={numChannels:t};return b.runKernel(Hn,w,y)}const[l,h]=r?[e.videoWidth,e.videoHeight]:[e.width,e.height];let u;if(a)u=e.getContext("2d").getImageData(0,0,l,h).data;else if(s||n)u=e.data;else if(o||r||i){if(Dt==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")Dt=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else Dt=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});Dt.canvas.width=l,Dt.canvas.height=h,Dt.drawImage(e,0,0,l,h),u=Dt.getImageData(0,0,l,h).data}let f;if(t===4)f=new Int32Array(u);else{const w=l*h;f=new Int32Array(w*t);for(let y=0;y<w;y++)for(let $=0;$<t;++$)f[y*t+$]=u[y*4+$]}return jf(f,[h,l,t],"int32")}function Pg(e){if(e.rank!==2&&e.rank!==3)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${e.rank}.`);const t=e.rank===2?1:e.shape[2];if(t>4||t===2)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${t}`);if(e.dtype!=="float32"&&e.dtype!=="int32")throw new Error(`Unsupported type for toPixels: ${e.dtype}. Please use float32 or int32 tensors.`)}async function Ow(e,t){let n=d(e,"img","toPixels");if(!(e instanceof Q)){const l=n;n=j(l,"int32"),l.dispose()}Pg(n);const[s,r]=n.shape.slice(0,2),o=n.rank===2?1:n.shape[2],a=await n.data(),i=n.dtype==="float32"?255:1,c=new Uint8ClampedArray(r*s*4);for(let l=0;l<s*r;++l){const h=[0,0,0,255];for(let f=0;f<o;f++){const g=a[l*o+f];if(n.dtype==="float32"){if(g<0||g>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${g}.`)}else if(n.dtype==="int32"&&(g<0||g>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${g}.`);o===1?(h[0]=g*i,h[1]=g*i,h[2]=g*i):h[f]=g*i}const u=l*4;c[u+0]=Math.round(h[0]),c[u+1]=Math.round(h[1]),c[u+2]=Math.round(h[2]),c[u+3]=Math.round(h[3])}if(t!=null){ds||ve(So,b.backendName)!=null&&(console.warn("tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead."),ds=!0),t.width=r,t.height=s;const l=t.getContext("2d"),h=new ImageData(c,r,s);l.putImageData(h,0,0)}return n!==e&&n.dispose(),c}const Lw=m({fromPixels_:Rg});function Ww(e,t){const n=e.shape.length,s=t.shape.length;if(n<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(s<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${s}.`);if(t.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[s-1]>n)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[s-1]} vs. ${n}`);if(H(e.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);const r=t.shape,o=r[r.length-1];let a=1;for(let u=0;u<r.length-1;++u)a*=r[u];const i=e.shape,c=r.slice();c.pop();let l=1;for(let u=o;u<n;++u)l*=i[u],c.push(i[u]);const h=[...ge(e.shape).map(u=>u/l),1].slice(0,o);return[c,a,l,h]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pn=-2,Og=-1;function Kw(e,t,n){const s=e.shape.length;p(s===t.length,()=>`Error in slice${s}D: Length of begin ${t} must match the rank of the array (${s}).`),p(s===n.length,()=>`Error in slice${s}D: Length of size ${n} must match the rank of the array (${s}).`);for(let r=0;r<s;++r)p(t[r]+n[r]<=e.shape[r],()=>`Error in slice${s}D: begin[${r}] + size[${r}] (${t[r]+n[r]}) would overflow input.shape[${r}] (${e.shape[r]})`)}function Uw(e,t,n){const s=[];for(let r=0;r<e.length;r++)s[r]=Math.ceil((t[r]-e[r])/n[r]);return s}function Gw(e,t,n){let s=n.length;for(let r=0;r<n.length;r++)if(n[r]>1){s=r;break}for(let r=s+1;r<n.length;r++)if(t[r]>0||n[r]!==e[r])return!1;return!0}function zw(e,t){let n=e.length>0?e[e.length-1]:1;for(let s=0;s<e.length-1;s++)n+=e[s]*t[s];return n}function qw(e,t,n){let s;const r=e.shape.length;typeof t=="number"?s=[t,...new Array(r-1).fill(0)]:t.length<r?s=t.concat(new Array(r-t.length).fill(0)):s=t.slice(),s.forEach(a=>{p(a!==-1,()=>"slice() does not support negative begin indexing.")});let o;return n==null?o=new Array(r).fill(-1):typeof n=="number"?o=[n,...new Array(r-1).fill(-1)]:n.length<r?o=n.concat(new Array(r-n.length).fill(-1)):o=n,o=o.map((a,i)=>a>=0?a:(p(a===-1,()=>`Negative size values should be exactly -1 but got ${a} for the slice() size at index ${i}.`),e.shape[i]-s[i])),[s,o]}function Hw(e,t,n,s,r,o,a,i,c){let l;if(s==null?(l=new Array(t.length),l.fill(1)):l=s,a!=null&&a&a-1)throw new Error("Multiple ellipses in slice is not allowed.");let h=!1;const u={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:r,endMask:o,ellipsisMask:a,newAxisMask:i,shrinkAxisMask:c};for(let k=0;k<u.dims;k++)h&&1<<k&i&&u.numAddAxisAfterEllipsis++,1<<k&a&&(h=!0);h||(u.ellipsisMask|=1<<u.dims,u.dims++);const f={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};Lg(u,f);let g=!0,w=!0,y=!0;const $=[],E=[];for(let k=0;k<e.length;++k){if(f.strides[k]===0)throw Error(`strides[${k}] must be non-zero`);const v=!!(f.shrinkAxisMask&1<<k),S=e[k];if(S===-1){$.push(v?1:-1);continue}const T=[f.beginMask&1<<k,f.endMask&1<<k],N=[f.strides[k]>0?0:-1,f.strides[k]>0?S:S-1];if(v&&f.strides[k]<=0)throw Error("only stride 1 allowed on non-range indexing.");y=y&&f.strides[k]===1;const D=!!(f.beginMask&1<<k&&f.endMask&1<<k);if(f.beginValid&&f.endValid){if(v){const P=f.begin[k]<0?S+f.begin[k]:f.begin[k];if(f.begin[k]=P,f.end[k]=f.begin[k]+1,P<0||P>=S)throw Error(`slice index ${f.begin[k]} of dimension ${k} out of bounds.`)}else f.begin[k]=ps(f.begin[k],0,f.strides[k],S,T,N),f.end[k]=ps(f.end[k],1,f.strides[k],S,T,N);const C=f.strides[k]===1&&f.begin[k]===0&&f.end[k]===S;g=g&&C,w=w&&(k===0&&f.strides[k]===1||C)}else g=g&&f.strides[k]===1&&D,w=w&&(k===0&&f.strides[k]===1||D);let A,B=!1;if(f.beginValid&&f.endValid?(A=f.end[k]-f.begin[k],B=!0):v?(A=1,B=!0):D&&S>=0&&(f.strides[k]<0?A=-S:A=S,B=!0),B){let C;A===0||A<0!=f.strides[k]<0?C=0:C=Math.trunc(A/f.strides[k])+(A%f.strides[k]!==0?1:0),$.push(C)}else $.push(-1)}for(let k=0;k<f.finalShapeGatherIndices.length;++k){const v=f.finalShapeGatherIndices[k];v>=0?E.push($[v]):v===pn&&E.push(1)}return{finalShapeSparse:E.filter((k,v)=>f.finalShapeGatherIndices[v]!==pn),finalShape:E,isIdentity:g,sliceDim0:w,isSimpleSlice:y,begin:f.begin,end:f.end,strides:f.strides}}function Lg(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=new Array(t.dims),t.end=new Array(t.dims),t.strides=new Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=new Array(t.dims);for(let s=0;s<e.dims;s++)if(1<<s&e.ellipsisMask){const r=Math.min(t.dims-(e.dims-s)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<r;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=s}else if(1<<s&e.newAxisMask)t.finalShapeGatherIndices.push(pn),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[s]),e.end!=null&&(t.end[n]=e.end[s]),t.strides[n]=e.strides[s],e.beginMask&1<<s&&(t.beginMask|=1<<n),e.endMask&1<<s&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<s?(t.finalShapeGatherIndices.push(Og),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(s)),t.inputShapeGatherIndicesSparse[n]=s,n++}}function ps(e,t,n,s,r,o){if(r[t])return n>0?o[t]:o[t+1&1];{const a=e<0?s+e:e;return a<o[0]?o[0]:a>o[1]?o[1]:a}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vw(e,t){const n=e[0].length;e.forEach((r,o)=>{p(r.length===n,()=>`Error in concat${n}D: rank of tensors[${o}] must be the same as the rank of the rest (${n})`)}),p(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);const s=e[0];e.forEach((r,o)=>{for(let a=0;a<n;a++)p(a===t||r[a]===s[a],()=>`Error in concat${n}D: Shape of tensors[${o}] (${r}) does not match the shape of the rest (${s}) along the non-concatenated axis ${o}.`)})}function jw(e,t){const n=e[0].slice();for(let s=1;s<e.length;s++)n[t]+=e[s][t];return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xw(e,t,n){const s=n*(typeof e=="number"?e:e[0]),r=t*(typeof e=="number"?e:e[1]);return[s,r]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jw(e,t,n,s=!0){let r=[];if(s)r=r.concat(t.slice(0)),r.push(e[0]/n),r=r.concat(e.slice(1));else{r=r.concat(e[0]);const o=t.length;for(let a=0;a<o;++a)r=r.concat([e[a+1]/t[a],t[a]]);r=r.concat(e.slice(o+1))}return r}function Yw(e,t,n=!0){const s=[];if(n){s.push(t);for(let r=t+1;r<e;++r)r<=2*t?(s.push(r),s.push(r-(t+1))):s.push(r)}else{const r=[],o=[];for(let a=1;a<e;++a)a>=t*2+1||a%2===1?o.push(a):r.push(a);s.push(...r),s.push(0),s.push(...o)}return s}function Zw(e,t,n,s=!0){const r=[];s?r.push(e[0]/n):r.push(e[0]*n);for(let o=1;o<e.length;++o)o<=t.length?s?r.push(t[o-1]*e[o]):r.push(e[o]/t[o-1]):r.push(e[o]);return r}function Qw(e,t){const n=[0];for(let s=0;s<t;++s)n.push(e[s][0]);return n}function ty(e,t,n){const s=e.slice(0,1);for(let r=0;r<n;++r)s.push(e[r+1]-t[r][0]-t[r][1]);return s}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ey=1.7580993408473768,ny=1.0507009873554805;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sy=.3275911,ry=.254829592,oy=-.284496736,ay=1.421413741,iy=-1.453152027,cy=1.061405429;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ly(e,t){if(e.length!==t.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);const n=new Float32Array(e.length*2);for(let s=0;s<n.length;s+=2)n[s]=e[s/2],n[s+1]=t[s/2];return n}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const je="->",Wg=/->/g,gs=",",ms="...";function uy(e,t){e=e.replace(/\s/g,"");const n=(e.length-e.replace(Wg,"").length)/je.length;if(n<1)throw new Error("Equations without an arrow are not supported.");if(n>1)throw new Error(`Equation must contain exactly one arrow ("${je}").`);const[s,r]=e.split(je);p(s.indexOf(ms)===-1,()=>`The ellipsis notation ("${ms}") is not supported yet.`);const o=s.split(gs),a=o.length;if(t!==a)throw new Error(`Expected ${a} input tensors, received ${t}`);if(a>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const i=[];for(let f=0;f<r.length;++f){const g=r[f];if(!o.some(w=>w.indexOf(g)!==-1))throw new Error(`Output subscripts contain the label ${g} not present in the input subscripts.`);i.indexOf(g)===-1&&i.push(g)}for(let f=0;f<s.length;++f){const g=s[f];i.indexOf(g)===-1&&g!==gs&&i.push(g)}const c=new Array(o.length);for(let f=0;f<a;++f){if(new Set(o[f].split("")).size!==o[f].length)throw new Error(`Found duplicate axes in input component ${o[f]}. Support for duplicate axes in input is not implemented yet.`);c[f]=[];for(let g=0;g<o[f].length;++g)c[f].push(i.indexOf(o[f][g]))}const l=i.length,h=r.length,u=[];for(let f=h;f<l;++f)u.push(f);return{allDims:i,summedDims:u,idDims:c}}function hy(e,t){let n=new Array(e);n.fill(-1);for(let r=0;r<t.length;++r)n[t[r]]=r;const s=[];for(let r=0;r<e;++r)n[r]===-1&&s.push(r);return n=n.filter(r=>r!==-1),{permutationIndices:n,expandDims:s}}function fy(e,t,n){const s=new Array(e);for(let r=0;r<n.length;++r){const o=n[r].shape;for(let a=0;a<t[r].length;++a)s[t[r][a]]===void 0?s[t[r][a]]=o[a]:p(s[t[r][a]]===o[a],()=>`Expected dimension ${s[t[r][a]]} at axis ${a} of input shaped ${JSON.stringify(o)}, but got dimension ${o[a]}`)}}function dy(e,t){const n=e,s=[];let r=0;e.length===0&&n.push(-1),r=e.length+1;for(let a=0;a<r;++a)s.push([]);const o=[];for(let a=0;a<n.length;++a){const i=n[a],c=Kg(t,i);for(const l of c)o.indexOf(l)===-1&&(s[a].push(l),o.push(l))}return{path:n,steps:s}}function py(e){return e.every((t,n)=>t===n)}function Kg(e,t){const n=[];for(let s=0;s<e.length;++s)(e[s].length===0||e[s].indexOf(t)!==-1||t===-1)&&n.push(s);return n}function gy(e,t,n=0){let s=[];if(typeof t=="number")p(e.shape[n]%t===0,()=>"Number of splits must evenly divide the axis."),s=new Array(t).fill(e.shape[n]/t);else{const r=t.reduce((a,i)=>(i===-1&&(a+=1),a),0);p(r<=1,()=>"There should be only one negative value in split array.");const o=t.indexOf(-1);if(o!==-1){const a=t.reduce((i,c)=>c>0?i+c:i);t[o]=e.shape[n]-a}p(e.shape[n]===t.reduce((a,i)=>a+i),()=>"The sum of sizes must match the size of the axis dimension."),s=t}return s}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function my(e,t,n){const s=[],r=e.length;for(let o=0;o<r;o++)o!==t?s.push(e[o]):s.push(n);return s}function by(e,t,n,s){const r=t.shape.length,o=e.shape.length;if(s!==0&&(s<-r||s>r))throw new Error(`Expect batchDims in the range of [-${r}, ${r}], but got ${s}`);if(s<0&&(s+=r),s>o)throw new Error(`batchDims (${s}) must be less than rank(x) (
    ${o}).`);if(n<s)throw new Error(`batchDims (${s}) must be less than or equal to axis (${n}).`);for(let u=0;u<s;++u)if(e.shape[u]!==t.shape[u])throw new Error(`x.shape[${u}]: ${e.shape[u]} should be equal to indices.shape[${u}]: ${t.shape[u]}.`);const a=e.shape[n],i=[];let c=1,l=1,h=1;for(let u=0;u<s;++u)i.push(e.shape[u]),c*=e.shape[u];for(let u=s;u<n;u++)i.push(e.shape[u]),l*=e.shape[u];for(let u=s;u<r;u++)i.push(t.shape[u]);for(let u=n+1;u<o;u++)i.push(e.shape[u]),h*=e.shape[u];return{batchSize:c,sliceSize:h,outerSize:l,dimSize:a,outputShape:i}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wy(e){try{return e.map(t=>Ye(t))}catch(t){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${t}`)}}function yy(e){return e.map(t=>$n(t))}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */wg();export{Zm as $,El as A,Fm as B,xw as C,Rm as D,He as E,Et as F,j as G,Pm as H,Om as I,Ct as J,rt as K,Lm as L,Wm as M,Km as N,zi as O,Um as P,Gm as Q,vn as R,zm as S,qm as T,Hm as U,Vm as V,jm as W,cr as X,Xm as Y,Jm as Z,Ym as _,Xt as a,Ib as a$,Js as a0,Qm as a1,tb as a2,G as a3,nb as a4,sb as a5,Tw as a6,re as a7,ou as a8,pd as a9,is as aA,Qs as aB,Cw as aC,mb as aD,bb as aE,_e as aF,Ku as aG,wb as aH,yb as aI,Ju as aJ,ln as aK,Qu as aL,eh as aM,$b as aN,Bw as aO,kb as aP,L as aQ,le as aR,oh as aS,xb as aT,Eb as aU,lh as aV,un as aW,vb as aX,as as aY,hn as aZ,Sb as a_,rb as aa,Zl as ab,ob as ac,hb as ad,Qt as ae,At as af,fb as ag,xu as ah,sr as ai,Oe as aj,vu as ak,Rc as al,Aw as am,Iu as an,Iw as ao,In as ap,Au as aq,fn as ar,Tn as as,Mw as at,Dw as au,Pf as av,db as aw,pb as ax,gb as ay,Fu as az,it as b,qu as b$,Tb as b0,Ew as b1,I as b2,Db as b3,Ab as b4,wt as b5,Sn as b6,wh as b7,_b as b8,Vt as b9,pf as bA,x as bB,te as bC,Vb as bD,jb as bE,Xb as bF,Jb as bG,rr as bH,kf as bI,Yb as bJ,W as bK,vw as bL,tr as bM,Zb as bN,Qb as bO,tw as bP,ce as bQ,ew as bR,Nw as bS,nw as bT,sw as bU,q as bV,rw as bW,ow as bX,aw as bY,iw as bZ,cw as b_,Nb as ba,m as bb,Mb as bc,We as bd,Cb as be,Bb as bf,Fb as bg,Rb as bh,Pb as bi,Ae as bj,Ch as bk,Mc as bl,Ob as bm,Lb as bn,Wb as bo,Kb as bp,Ub as bq,Gb as br,of as bs,zb as bt,nr as bu,qb as bv,Ne as bw,Me as bx,Hb as by,Rn as bz,Ot as c,ey as c$,Dh as c0,Fw as c1,Sw as c2,_w as c3,Ce as c4,vt as c5,gt as c6,Kf as c7,or as c8,Be as c9,pe as cA,ct as cB,p as cC,Vi as cD,X as cE,Lt as cF,bn as cG,Pw as cH,Qi as cI,Q as cJ,st as cK,eb as cL,ge as cM,kr as cN,Gg as cO,dm as cP,Xg as cQ,ly as cR,H as cS,Ye as cT,he as cU,qg as cV,gn as cW,$n as cX,vr as cY,gm as cZ,ny as c_,qf as ca,lw as cb,Rw as cc,F as cd,U as ce,uw as cf,os as cg,ie as ch,ft as ci,xe as cj,jf as ck,hw as cl,fw as cm,dw as cn,gw as co,ke as cp,mw as cq,cs as cr,bw as cs,ww as ct,yw as cu,ir as cv,$w as cw,kw as cx,Mt as cy,rd as cz,mm as d,Zr as d$,sy as d0,ry as d1,oy as d2,ay as d3,iy as d4,cy as d5,Fo as d6,Re as d7,bs as d8,Ra as d9,Fe as dA,cb as dB,ub as dC,ib as dD,Le as dE,fm as dF,Wr as dG,Kr as dH,Ur as dI,Gr as dJ,zr as dK,qr as dL,Hr as dM,jr as dN,Vr as dO,oa as dP,ua as dQ,Xr as dR,Zc as dS,Jr as dT,Am as dU,Qg as dV,Zg as dW,Yr as dX,ja as dY,qw as dZ,Kw as d_,Vg as da,Y as db,Vn as dc,Hg as dd,vs as de,ro as df,_s as dg,wy as dh,Yg as di,Jg as dj,yn as dk,hm as dl,ab as dm,mn as dn,Ze as dp,Gw as dq,zw as dr,yy as ds,zg as dt,Rr as du,Pr as dv,Or as dw,xs as dx,Lr as dy,Ue as dz,R as e,Hn as e$,Jw as e0,Yw as e1,Zw as e2,Qw as e3,ty as e4,Qr as e5,eo as e6,wa as e7,Ca as e8,Es as e9,ko as eA,xo as eB,Eo as eC,vo as eD,Dm as eE,nm as eF,em as eG,So as eH,ma as eI,ei as eJ,To as eK,uy as eL,fy as eM,dy as eN,hy as eO,py as eP,Do as eQ,sm as eR,_o as eS,Ao as eT,No as eU,Mo as eV,Co as eW,Bo as eX,Ro as eY,Po as eZ,Oo as e_,jg as ea,no as eb,so as ec,oo as ed,jw as ee,qo as ef,ao as eg,Vw as eh,io as ei,rl as ej,Pe as ek,co as el,lo as em,uo as en,Qc as eo,tm as ep,ho as eq,fo as er,po as es,bo as et,lb as eu,go as ev,mo as ew,wo as ex,yo as ey,$o as ez,bm as f,Ka as f$,Lo as f0,jn as f1,Xn as f2,Tt as f3,Ko as f4,Ww as f5,Wo as f6,by as f7,Uo as f8,Go as f9,ga as fA,ba as fB,ya as fC,Jd as fD,ka as fE,Zd as fF,Ea as fG,vi as fH,xa as fI,va as fJ,ut as fK,Sa as fL,Ia as fM,Ta as fN,Da as fO,Ma as fP,Io as fQ,Ba as fR,Fa as fS,La as fT,Oa as fU,cm as fV,Pa as fW,im as fX,Wa as fY,Ii as fZ,Xw as f_,zo as fa,Ho as fb,Vo as fc,jo as fd,Xo as fe,Jo as ff,Yo as fg,Zo as fh,Qo as fi,ta as fj,ea as fk,na as fl,sa as fm,ra as fn,rm as fo,aa as fp,ia as fq,ca as fr,am as fs,om as ft,la as fu,ha as fv,fa as fw,da as fx,pa as fy,ri as fz,M as g,Ua as g0,Ga as g1,pw as g2,qa as g3,Ha as g4,Va as g5,Za as g6,Ya as g7,Xa as g8,Ja as g9,ji as gA,Ow as gB,Lw as gC,Xi as gD,Ug as gE,pm as gF,Qa as ga,ni as gb,ii as gc,ci as gd,Ss as ge,li as gf,si as gg,gy as gh,ti as gi,lm as gj,ui as gk,Si as gl,fi as gm,Hw as gn,Uw as go,di as gp,mi as gq,bi as gr,wi as gs,za as gt,yi as gu,$i as gv,xi as gw,Ei as gx,my as gy,um as gz,wm as h,ym as i,$m as j,km as k,xm as l,Em as m,vm as n,Sm as o,Im as p,Tm as q,il as r,_m as s,zn as t,Nm as u,En as v,Mm as w,Cm as x,Bm as y,ml as z};
