(function(){var k;function n(a,b){function c(){}
c.prototype=b.prototype;a.o=b.prototype;a.prototype=new c;a.prototype.constructor=a;for(var d in b)if(Object.defineProperties){var e=Object.getOwnPropertyDescriptor(b,d);e&&Object.defineProperty(a,d,e)}else a[d]=b[d]}
for(var aa="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){a!=Array.prototype&&a!=Object.prototype&&(a[b]=c.value)},ba="function"==typeof Object.create?Object.create:function(a){function b(){}
b.prototype=a;return new b},ca="undefined"!=typeof Reflect&&Reflect.construct||function(a,b,c){void 0===c&&(c=a);
c=ba(c.prototype||Object.prototype);return Function.prototype.apply.call(a,c,b)||c},da="undefined"!=typeof window&&window===this?this:"undefined"!=typeof global&&null!=global?global:this,ea=["Reflect",
"construct"],fa=0;fa<ea.length-1;fa++){var ha=ea[fa];ha in da||(da[ha]={});da=da[ha]}var ia=ea[ea.length-1],ja=da[ia],ka;ka=ja||ca;ka!=ja&&null!=ka&&aa(da,ia,{configurable:!0,writable:!0,value:ka});var p=this;function q(a){return void 0!==a}
function r(a){return"string"==typeof a}
function t(a,b,c){a=a.split(".");c=c||p;a[0]in c||!c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)!a.length&&q(b)?c[d]=b:c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}}
function u(a,b){for(var c=a.split("."),d=b||p,e;e=c.shift();)if(null!=d[e])d=d[e];else return null;return d}
function la(){}
function ma(a){a.ja=void 0;a.getInstance=function(){return a.ja?a.ja:a.ja=new a}}
function na(a){var b=typeof a;if("object"==b)if(a){if(a instanceof Array)return"array";if(a instanceof Object)return b;var c=Object.prototype.toString.call(a);if("[object Window]"==c)return"object";if("[object Array]"==c||"number"==typeof a.length&&"undefined"!=typeof a.splice&&"undefined"!=typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("splice"))return"array";if("[object Function]"==c||"undefined"!=typeof a.call&&"undefined"!=typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("call"))return"function"}else return"null";
else if("function"==b&&"undefined"==typeof a.call)return"object";return b}
function oa(a){return"array"==na(a)}
function pa(a){var b=na(a);return"array"==b||"object"==b&&"number"==typeof a.length}
function qa(a){return"function"==na(a)}
function ra(a){var b=typeof a;return"object"==b&&null!=a||"function"==b}
var sa="closure_uid_"+(1E9*Math.random()>>>0),ta=0;function ua(a,b,c){return a.call.apply(a.bind,arguments)}
function va(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var c=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(c,d);return a.apply(b,c)}}return function(){return a.apply(b,arguments)}}
function v(a,b,c){Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?v=ua:v=va;return v.apply(null,arguments)}
function w(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var b=c.slice();b.push.apply(b,arguments);return a.apply(this,b)}}
var y=Date.now||function(){return+new Date};
function z(a,b){function c(){}
c.prototype=b.prototype;a.o=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.gb=function(a,c,f){for(var d=Array(arguments.length-2),e=2;e<arguments.length;e++)d[e-2]=arguments[e];return b.prototype[c].apply(a,d)}}
;var wa=document,A=window;function B(a){if(Error.captureStackTrace)Error.captureStackTrace(this,B);else{var b=Error().stack;b&&(this.stack=b)}a&&(this.message=String(a))}
z(B,Error);B.prototype.name="CustomError";var xa=String.prototype.trim?function(a){return a.trim()}:function(a){return a.replace(/^[\s\xa0]+|[\s\xa0]+$/g,"")};
function ya(a,b){return a<b?-1:a>b?1:0}
function Aa(a){for(var b=0,c=0;c<a.length;++c)b=31*b+a.charCodeAt(c)>>>0;return b}
;var Ba=Array.prototype.indexOf?function(a,b,c){return Array.prototype.indexOf.call(a,b,c)}:function(a,b,c){c=null==c?0:0>c?Math.max(0,a.length+c):c;
if(r(a))return r(b)&&1==b.length?a.indexOf(b,c):-1;for(;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},C=Array.prototype.forEach?function(a,b,c){Array.prototype.forEach.call(a,b,c)}:function(a,b,c){for(var d=a.length,e=r(a)?a.split(""):a,f=0;f<d;f++)f in e&&b.call(c,e[f],f,a)},Ca=Array.prototype.map?function(a,b,c){return Array.prototype.map.call(a,b,c)}:function(a,b,c){for(var d=a.length,e=Array(d),f=r(a)?a.split(""):a,g=0;g<d;g++)g in f&&(e[g]=b.call(c,f[g],g,a));
return e};
function Da(a,b){a:{var c=a.length;for(var d=r(a)?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){c=e;break a}c=-1}return 0>c?null:r(a)?a.charAt(c):a[c]}
function Ea(a,b){var c=Ba(a,b);0<=c&&Array.prototype.splice.call(a,c,1)}
function Fa(a){var b=a.length;if(0<b){for(var c=Array(b),d=0;d<b;d++)c[d]=a[d];return c}return[]}
function Ga(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(pa(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;function Ha(a,b){this.b=q(a)?a:0;this.f=q(b)?b:0}
Ha.prototype.equals=function(a){return a instanceof Ha&&(this==a?!0:this&&a?this.b==a.b&&this.f==a.f:!1)};
Ha.prototype.ceil=function(){this.b=Math.ceil(this.b);this.f=Math.ceil(this.f);return this};
Ha.prototype.floor=function(){this.b=Math.floor(this.b);this.f=Math.floor(this.f);return this};
Ha.prototype.round=function(){this.b=Math.round(this.b);this.f=Math.round(this.f);return this};function Ia(a,b){this.width=a;this.height=b}
k=Ia.prototype;k.aspectRatio=function(){return this.width/this.height};
k.isEmpty=function(){return!(this.width*this.height)};
k.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
k.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
k.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};function Ja(a,b){for(var c=pa(b),d=c?b:arguments,c=c?0:1;c<d.length&&(a=a[d[c]],q(a));c++);return a}
function Ka(a){var b=La,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function Ma(a){for(var b in a)return!1;return!0}
function Na(a,b){if(null!==a&&b in a)throw Error('The object already contains the key "'+b+'"');a[b]=!0}
function Oa(a){var b={},c;for(c in a)b[c]=a[c];return b}
var Pa="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function Qa(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<Pa.length;f++)c=Pa[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;function Sa(a){Sa[" "](a);return a}
Sa[" "]=la;function Ta(a,b){var c=Ua;return Object.prototype.hasOwnProperty.call(c,a)?c[a]:c[a]=b(a)}
;function Va(){var a=Wa;try{var b;if(b=!!a&&null!=a.location.href)a:{try{Sa(a.foo);b=!0;break a}catch(c){}b=!1}return b}catch(c){return!1}}
;var Xa=function(){var a=!1;try{var b=Object.defineProperty({},"passive",{get:function(){a=!0}});
p.addEventListener("test",null,b)}catch(c){}return a}();var Ya=!1,Za="";function $a(a){a=a.match(/[\d]+/g);if(!a)return"";a.length=3;return a.join(".")}
(function(){if(navigator.plugins&&navigator.plugins.length){var a=navigator.plugins["Shockwave Flash"];if(a&&(Ya=!0,a.description)){Za=$a(a.description);return}if(navigator.plugins["Shockwave Flash 2.0"]){Ya=!0;Za="2.0.0.11";return}}if(navigator.mimeTypes&&navigator.mimeTypes.length&&(a=navigator.mimeTypes["application/x-shockwave-flash"],Ya=!(!a||!a.enabledPlugin))){Za=$a(a.enabledPlugin.description);return}try{var b=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7");Ya=!0;Za=$a(b.GetVariable("$version"));
return}catch(c){}try{b=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");Ya=!0;Za="6.0.21";return}catch(c){}try{b=new ActiveXObject("ShockwaveFlash.ShockwaveFlash"),Ya=!0,Za=$a(b.GetVariable("$version"))}catch(c){}})();
var ab=Ya,bb=Za;var E;a:{var cb=p.navigator;if(cb){var db=cb.userAgent;if(db){E=db;break a}}E=""}function F(a){return-1!=E.indexOf(a)}
;function eb(){return(F("Chrome")||F("CriOS"))&&!F("Edge")}
;function fb(){return F("iPhone")&&!F("iPod")&&!F("iPad")}
;var gb=F("Opera"),G=F("Trident")||F("MSIE"),hb=F("Edge"),ib=F("Gecko")&&!(-1!=E.toLowerCase().indexOf("webkit")&&!F("Edge"))&&!(F("Trident")||F("MSIE"))&&!F("Edge"),jb=-1!=E.toLowerCase().indexOf("webkit")&&!F("Edge"),kb=F("Macintosh"),lb=F("Windows"),mb=F("Android"),nb=fb(),ob=F("iPad"),pb=F("iPod");function qb(){var a=p.document;return a?a.documentMode:void 0}
var rb;a:{var sb="",tb=function(){var a=E;if(ib)return/rv\:([^\);]+)(\)|;)/.exec(a);if(hb)return/Edge\/([\d\.]+)/.exec(a);if(G)return/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a);if(jb)return/WebKit\/(\S+)/.exec(a);if(gb)return/(?:Version)[ \/]?(\S+)/.exec(a)}();
tb&&(sb=tb?tb[1]:"");if(G){var ub=qb();if(null!=ub&&ub>parseFloat(sb)){rb=String(ub);break a}}rb=sb}var vb=rb,Ua={};
function wb(a){return Ta(a,function(){for(var b=0,c=xa(String(vb)).split("."),d=xa(String(a)).split("."),e=Math.max(c.length,d.length),f=0;0==b&&f<e;f++){var g=c[f]||"",h=d[f]||"";do{g=/(\d*)(\D*)(.*)/.exec(g)||["","","",""];h=/(\d*)(\D*)(.*)/.exec(h)||["","","",""];if(0==g[0].length&&0==h[0].length)break;b=ya(0==g[1].length?0:parseInt(g[1],10),0==h[1].length?0:parseInt(h[1],10))||ya(0==g[2].length,0==h[2].length)||ya(g[2],h[2]);g=g[3];h=h[3]}while(0==b)}return 0<=b})}
var xb;var yb=p.document;xb=yb&&G?qb()||("CSS1Compat"==yb.compatMode?parseInt(vb,10):5):void 0;(function(){if(lb){var a=/Windows NT ([0-9.]+)/;return(a=a.exec(E))?a[1]:"0"}return kb?(a=/10[_.][0-9_.]+/,(a=a.exec(E))?a[0].replace(/_/g,"."):"10"):mb?(a=/Android\s+([^\);]+)(\)|;)/,(a=a.exec(E))?a[1]:""):nb||ob||pb?(a=/(?:iPhone|CPU)\s+OS\s+(\S+)/,(a=a.exec(E))?a[1].replace(/_/g,"."):""):""})();var zb=F("Firefox"),Ab=fb()||F("iPod"),Bb=F("iPad"),Cb=F("Android")&&!(eb()||F("Firefox")||F("Opera")||F("Silk")),Db=eb(),Eb=F("Safari")&&!(eb()||F("Coast")||F("Opera")||F("Edge")||F("Silk")||F("Android"))&&!(fb()||F("iPad")||F("iPod"));function Fb(a){return(a=a.exec(E))?a[1]:""}
(function(){if(zb)return Fb(/Firefox\/([0-9.]+)/);if(G||hb||gb)return vb;if(Db)return fb()||F("iPad")||F("iPod")?Fb(/CriOS\/([0-9.]+)/):Fb(/Chrome\/([0-9.]+)/);if(Eb&&!(fb()||F("iPad")||F("iPod")))return Fb(/Version\/([0-9.]+)/);if(Ab||Bb){var a=/Version\/(\S+).*Mobile\/(\S+)/.exec(E);if(a)return a[1]+"."+a[2]}else if(Cb)return(a=Fb(/Android\s+([0-9.]+)/))?a:Fb(/Version\/([0-9.]+)/);return""})();!ib&&!G||G&&9<=Number(xb)||ib&&wb("1.9.1");G&&wb("9");function Gb(){this.b="";this.f=Hb}
Gb.prototype.ia=!0;Gb.prototype.ha=function(){return this.b};
function Ib(a){return a instanceof Gb&&a.constructor===Gb&&a.f===Hb?a.b:"type_error:TrustedResourceUrl"}
var Hb={};function H(){this.b="";this.f=Jb}
H.prototype.ia=!0;H.prototype.ha=function(){return this.b};
function Kb(a){return a instanceof H&&a.constructor===H&&a.f===Jb?a.b:"type_error:SafeUrl"}
var Lb=/^(?:(?:https?|mailto|ftp):|[^:/?#]*(?:[/?#]|$))/i;function Mb(a){if(a instanceof H)return a;a=a.ia?a.ha():String(a);Lb.test(a)||(a="about:invalid#zClosurez");return Nb(a)}
var Jb={};function Nb(a){var b=new H;b.b=a;return b}
Nb("about:blank");function Ob(){this.b=""}
Ob.prototype.ia=!0;Ob.prototype.ha=function(){return this.b};
function Pb(a){var b=new Ob;b.b=a;return b}
Pb("<!DOCTYPE html>");Pb("");Pb("<br>");function Qb(a,b){var c=b instanceof H?b:Mb(b);a.href=Kb(c)}
function Rb(a,b){a.src=Ib(b)}
;function Sb(a){var b=document;return r(a)?b.getElementById(a):a}
function Tb(a){if(!a)return null;if(a.firstChild)return a.firstChild;for(;a&&!a.nextSibling;)a=a.parentNode;return a?a.nextSibling:null}
function Ub(a){if(!a)return null;if(!a.previousSibling)return a.parentNode;for(a=a.previousSibling;a&&a.lastChild;)a=a.lastChild;return a}
function Vb(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;function Wb(a){Xb();var b=new Gb;b.b=a;return b}
var Xb=la;function Yb(a){"number"==typeof a&&(a=Math.round(a)+"px");return a}
;var Zb=/^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#([\s\S]*))?$/;function I(a){return a.match(Zb)}
function $b(a){return a?decodeURI(a):a}
function ac(a,b,c){if(oa(b))for(var d=0;d<b.length;d++)ac(a,String(b[d]),c);else null!=b&&c.push(a+(""===b?"":"="+encodeURIComponent(String(b))))}
function bc(a){var b=[],c;for(c in a)ac(c,a[c],b);return b.join("&")}
function cc(a,b){var c=bc(b);if(c){var d=a.indexOf("#");0>d&&(d=a.length);var e=a.indexOf("?");if(0>e||e>d){e=d;var f=""}else f=a.substring(e+1,d);d=[a.substr(0,e),f,a.substr(d)];e=d[1];d[1]=c?e?e+"&"+c:c:e;c=d[0]+(d[1]?"?"+d[1]:"")+d[2]}else c=a;return c}
;var dc=!!window.google_async_iframe_id,Wa=dc&&window.parent||window;function ec(a,b){var c=fc();this.label=a;this.type=b;this.value=c;this.duration=0;this.uniqueId=this.label+"_"+this.type+"_"+Math.random();this.slotId=void 0}
;function gc(a,b){this.events=[];this.f=b||p;var c=null;b&&(b.google_js_reporting_queue=b.google_js_reporting_queue||[],this.events=b.google_js_reporting_queue,c=b.google_measure_js_timing);a:{try{var d=(this.f||p).top.location.hash;if(d){var e=d.match(/\bdeid=([\d,]+)/);var f=e&&e[1]||"";break a}}catch(g){}f=""}f=f.indexOf&&0<=f.indexOf("1337");this.b=(this.b=null!=c?c:Math.random()<a)||f;c=this.f.performance;this.g=!!(c&&c.mark&&c.clearMarks&&f)}
gc.prototype.h=function(a){if(a&&this.g){var b=this.f.performance;b.clearMarks("goog_"+a.uniqueId+"_start");b.clearMarks("goog_"+a.uniqueId+"_end")}};
gc.prototype.start=function(a,b){if(!this.b)return null;var c=new ec(a,b);this.g&&this.f.performance.mark("goog_"+c.uniqueId+"_start");return c};
gc.prototype.end=function(a){this.b&&"number"==typeof a.value&&(a.duration=fc()-a.value,this.g&&this.f.performance.mark("goog_"+a.uniqueId+"_end"),this.b&&this.events.push(a))};
function fc(){var a=p.performance;return a&&a.now?a.now():y()}
;if(dc&&!Va()){var hc="."+wa.domain;try{for(;2<hc.split(".").length&&!Va();)wa.domain=hc=hc.substr(hc.indexOf(".")+1),Wa=window.parent}catch(a){}Va()||(Wa=window)}var ic=Wa,J=new gc(1,ic);function jc(){ic.google_measure_js_timing||(J.events!=J.f.google_js_reporting_queue&&(J.events.length=0,J.g&&C(J.events,J.h,J)),J.b=!1)}
if("complete"==ic.document.readyState)jc();else if(J.b){var kc=function(){jc()};
ic.addEventListener?ic.addEventListener("load",kc,Xa?void 0:!1):ic.attachEvent&&ic.attachEvent("onload",kc)};var lc=(new Date).getTime();function mc(a){if(!a)return"";a=a.split("#")[0].split("?")[0];a=a.toLowerCase();0==a.indexOf("//")&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");-1!=c&&(b=b.substring(0,c));a=a.substring(0,a.indexOf("://"));if("http"!==a&&"https"!==a&&"chrome-extension"!==a&&"file"!==a&&"android-app"!==a&&"chrome-search"!==a)throw Error("Invalid URI scheme in origin");var c="",d=b.indexOf(":");if(-1!=d){var e=b.substring(d+
1),b=b.substring(0,d);if("http"===a&&"80"!==e||"https"===a&&"443"!==e)c=":"+e}return a+"://"+b+c}
;function nc(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;x=m=0}
function b(a){for(var b=g,c=0;64>c;c+=4)b[c/4]=a[c]<<24|a[c+1]<<16|a[c+2]<<8|a[c+3];for(c=16;80>c;c++)a=b[c-3]^b[c-8]^b[c-14]^b[c-16],b[c]=(a<<1|a>>>31)&4294967295;a=e[0];for(var d=e[1],f=e[2],h=e[3],l=e[4],m,D,c=0;80>c;c++)40>c?20>c?(m=h^d&(f^h),D=1518500249):(m=d^f^h,D=1859775393):60>c?(m=d&f|h&(d|f),D=2400959708):(m=d^f^h,D=3395469782),m=((a<<5|a>>>27)&4294967295)+m+l+D+b[c]&4294967295,l=h,h=f,f=(d<<30|d>>>2)&4294967295,d=a,a=m;e[0]=e[0]+a&4294967295;e[1]=e[1]+d&4294967295;e[2]=e[2]+f&4294967295;
e[3]=e[3]+h&4294967295;e[4]=e[4]+l&4294967295}
function c(a,c){if("string"===typeof a){a=unescape(encodeURIComponent(a));for(var d=[],e=0,g=a.length;e<g;++e)d.push(a.charCodeAt(e));a=d}c||(c=a.length);d=0;if(0==m)for(;d+64<c;)b(a.slice(d,d+64)),d+=64,x+=64;for(;d<c;)if(f[m++]=a[d++],x++,64==m)for(m=0,b(f);d+64<c;)b(a.slice(d,d+64)),d+=64,x+=64}
function d(){var a=[],d=8*x;56>m?c(h,56-m):c(h,64-(m-56));for(var g=63;56<=g;g--)f[g]=d&255,d>>>=8;b(f);for(g=d=0;5>g;g++)for(var l=24;0<=l;l-=8)a[d++]=e[g]>>l&255;return a}
for(var e=[],f=[],g=[],h=[128],l=1;64>l;++l)h[l]=0;var m,x;a();return{reset:a,update:c,digest:d,ya:function(){for(var a=d(),b="",c=0;c<a.length;c++)b+="0123456789ABCDEF".charAt(Math.floor(a[c]/16))+"0123456789ABCDEF".charAt(a[c]%16);return b}}}
;function oc(a,b,c){var d=[],e=[];if(1==(oa(c)?2:1))return e=[b,a],C(d,function(a){e.push(a)}),pc(e.join(" "));
var f=[],g=[];C(c,function(a){g.push(a.key);f.push(a.value)});
c=Math.floor((new Date).getTime()/1E3);e=0==f.length?[c,b,a]:[f.join(":"),c,b,a];C(d,function(a){e.push(a)});
a=pc(e.join(" "));a=[c,a];0==g.length||a.push(g.join(""));return a.join("_")}
function pc(a){var b=nc();b.update(a);return b.ya().toLowerCase()}
;function qc(a){this.b=a||{cookie:""}}
k=qc.prototype;k.isEnabled=function(){return navigator.cookieEnabled};
k.set=function(a,b,c,d,e,f){if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');q(c)||(c=-1);e=e?";domain="+e:"";d=d?";path="+d:"";f=f?";secure":"";c=0>c?"":0==c?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(y()+1E3*c)).toUTCString();this.b.cookie=a+"="+b+e+d+c+f};
k.get=function(a,b){for(var c=a+"=",d=(this.b.cookie||"").split(";"),e=0,f;e<d.length;e++){f=xa(d[e]);if(0==f.lastIndexOf(c,0))return f.substr(c.length);if(f==a)return""}return b};
k.remove=function(a,b,c){var d=q(this.get(a));this.set(a,"",0,b,c);return d};
k.isEmpty=function(){return!this.b.cookie};
k.clear=function(){for(var a=(this.b.cookie||"").split(";"),b=[],c=[],d,e,f=0;f<a.length;f++)e=xa(a[f]),d=e.indexOf("="),-1==d?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));for(a=b.length-1;0<=a;a--)this.remove(b[a])};
var rc=new qc("undefined"==typeof document?null:document);rc.f=3950;function sc(){var a=[],b=mc(String(p.location.href)),c=p.__OVERRIDE_SID;null==c&&(c=(new qc(document)).get("SID"));if(c&&(b=(c=0==b.indexOf("https:")||0==b.indexOf("chrome-extension:"))?p.__SAPISID:p.__APISID,null==b&&(b=(new qc(document)).get(c?"SAPISID":"APISID")),b)){var c=c?"SAPISIDHASH":"APISIDHASH",d=String(p.location.href);return d&&b&&c?[c,oc(mc(d),b,a||null)].join(" "):null}return null}
;function tc(a,b,c){this.h=c;this.g=a;this.i=b;this.f=0;this.b=null}
tc.prototype.get=function(){if(0<this.f){this.f--;var a=this.b;this.b=a.next;a.next=null}else a=this.g();return a};
function uc(a,b){a.i(b);a.f<a.h&&(a.f++,b.next=a.b,a.b=b)}
;function vc(a){p.setTimeout(function(){throw a;},0)}
var wc;
function xc(){var a=p.MessageChannel;"undefined"===typeof a&&"undefined"!==typeof window&&window.postMessage&&window.addEventListener&&!F("Presto")&&(a=function(){var a=document.createElement("IFRAME");a.style.display="none";a.src="";document.documentElement.appendChild(a);var b=a.contentWindow,a=b.document;a.open();a.write("");a.close();var c="callImmediate"+Math.random(),d="file:"==b.location.protocol?"*":b.location.protocol+"//"+b.location.host,a=v(function(a){if(("*"==d||a.origin==d)&&a.data==
c)this.port1.onmessage()},this);
b.addEventListener("message",a,!1);this.port1={};this.port2={postMessage:function(){b.postMessage(c,d)}}});
if("undefined"!==typeof a&&!F("Trident")&&!F("MSIE")){var b=new a,c={},d=c;b.port1.onmessage=function(){if(q(c.next)){c=c.next;var a=c.oa;c.oa=null;a()}};
return function(a){d.next={oa:a};d=d.next;b.port2.postMessage(0)}}return"undefined"!==typeof document&&"onreadystatechange"in document.createElement("SCRIPT")?function(a){var b=document.createElement("SCRIPT");
b.onreadystatechange=function(){b.onreadystatechange=null;b.parentNode.removeChild(b);b=null;a();a=null};
document.documentElement.appendChild(b)}:function(a){p.setTimeout(a,0)}}
;function yc(){this.f=this.b=null}
var Ac=new tc(function(){return new zc},function(a){a.reset()},100);
yc.prototype.remove=function(){var a=null;this.b&&(a=this.b,this.b=this.b.next,this.b||(this.f=null),a.next=null);return a};
function zc(){this.next=this.scope=this.b=null}
zc.prototype.set=function(a,b){this.b=a;this.scope=b;this.next=null};
zc.prototype.reset=function(){this.next=this.scope=this.b=null};function Bc(a,b){Cc||Dc();Ec||(Cc(),Ec=!0);var c=Fc,d=Ac.get();d.set(a,b);c.f?c.f.next=d:c.b=d;c.f=d}
var Cc;function Dc(){if(-1!=String(p.Promise).indexOf("[native code]")){var a=p.Promise.resolve(void 0);Cc=function(){a.then(Gc)}}else Cc=function(){var a=Gc;
!qa(p.setImmediate)||p.Window&&p.Window.prototype&&!F("Edge")&&p.Window.prototype.setImmediate==p.setImmediate?(wc||(wc=xc()),wc(a)):p.setImmediate(a)}}
var Ec=!1,Fc=new yc;function Gc(){for(var a;a=Fc.remove();){try{a.b.call(a.scope)}catch(b){vc(b)}uc(Ac,a)}Ec=!1}
;function K(){this.f=this.f;this.F=this.F}
K.prototype.f=!1;K.prototype.dispose=function(){this.f||(this.f=!0,this.l())};
function Hc(a,b){a.f?q(void 0)?b.call(void 0):b():(a.F||(a.F=[]),a.F.push(q(void 0)?v(b,void 0):b))}
K.prototype.l=function(){if(this.F)for(;this.F.length;)this.F.shift()()};
function Ic(a){a&&"function"==typeof a.dispose&&a.dispose()}
function Jc(a){for(var b=0,c=arguments.length;b<c;++b){var d=arguments[b];pa(d)?Jc.apply(null,d):Ic(d)}}
;var Kc="StopIteration"in p?p.StopIteration:{message:"StopIteration",stack:""};function Lc(){}
Lc.prototype.next=function(){throw Kc;};
Lc.prototype.Y=function(){return this};
function Mc(a){if(a instanceof Lc)return a;if("function"==typeof a.Y)return a.Y(!1);if(pa(a)){var b=0,c=new Lc;c.next=function(){for(;;){if(b>=a.length)throw Kc;if(b in a)return a[b++];b++}};
return c}throw Error("Not implemented");}
function Nc(a,b){if(pa(a))try{C(a,b,void 0)}catch(c){if(c!==Kc)throw c;}else{a=Mc(a);try{for(;;)b.call(void 0,a.next(),void 0,a)}catch(c){if(c!==Kc)throw c;}}}
function Oc(a){if(pa(a))return Fa(a);a=Mc(a);var b=[];Nc(a,function(a){b.push(a)});
return b}
;function Pc(a){return/^\s*$/.test(a)?!1:/^[\],:{}\s\u2028\u2029]*$/.test(a.replace(/\\["\\\/bfnrtu]/g,"@").replace(/(?:"[^"\\\n\r\u2028\u2029\x00-\x08\x0a-\x1f]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)[\s\u2028\u2029]*(?=:|,|]|}|$)/g,"]").replace(/(?:^|:|,)(?:[\s\u2028\u2029]*\[)+/g,""))}
function Qc(a){a=String(a);if(Pc(a))try{return eval("("+a+")")}catch(b){}throw Error("Invalid JSON string: "+a);}
function Rc(a){var b=[];Sc(new Tc,a,b);return b.join("")}
function Tc(){}
function Sc(a,b,c){if(null==b)c.push("null");else{if("object"==typeof b){if(oa(b)){var d=b;b=d.length;c.push("[");for(var e="",f=0;f<b;f++)c.push(e),Sc(a,d[f],c),e=",";c.push("]");return}if(b instanceof String||b instanceof Number||b instanceof Boolean)b=b.valueOf();else{c.push("{");e="";for(d in b)Object.prototype.hasOwnProperty.call(b,d)&&(f=b[d],"function"!=typeof f&&(c.push(e),Uc(d,c),c.push(":"),Sc(a,f,c),e=","));c.push("}");return}}switch(typeof b){case "string":Uc(b,c);break;case "number":c.push(isFinite(b)&&
!isNaN(b)?String(b):"null");break;case "boolean":c.push(String(b));break;case "function":c.push("null");break;default:throw Error("Unknown type: "+typeof b);}}}
var Vc={'"':'\\"',"\\":"\\\\","/":"\\/","\b":"\\b","\f":"\\f","\n":"\\n","\r":"\\r","\t":"\\t","\x0B":"\\u000b"},Wc=/\uffff/.test("\uffff")?/[\\\"\x00-\x1f\x7f-\uffff]/g:/[\\\"\x00-\x1f\x7f-\xff]/g;function Uc(a,b){b.push('"',a.replace(Wc,function(a){var b=Vc[a];b||(b="\\u"+(a.charCodeAt(0)|65536).toString(16).substr(1),Vc[a]=b);return b}),'"')}
;function Xc(a){a.prototype.then=a.prototype.then;a.prototype.$goog_Thenable=!0}
function Yc(a){if(!a)return!1;try{return!!a.$goog_Thenable}catch(b){return!1}}
;function L(a,b){this.b=0;this.m=void 0;this.h=this.f=this.g=null;this.i=this.j=!1;if(a!=la)try{var c=this;a.call(b,function(a){Zc(c,2,a)},function(a){Zc(c,3,a)})}catch(d){Zc(this,3,d)}}
function $c(){this.next=this.context=this.f=this.g=this.b=null;this.h=!1}
$c.prototype.reset=function(){this.context=this.f=this.g=this.b=null;this.h=!1};
var ad=new tc(function(){return new $c},function(a){a.reset()},100);
function bd(a,b,c){var d=ad.get();d.g=a;d.f=b;d.context=c;return d}
function cd(a){return new L(function(b,c){c(a)})}
L.prototype.then=function(a,b,c){return dd(this,qa(a)?a:null,qa(b)?b:null,c)};
Xc(L);function ed(a,b){return dd(a,null,b,void 0)}
L.prototype.cancel=function(a){0==this.b&&Bc(function(){var b=new fd(a);gd(this,b)},this)};
function gd(a,b){if(0==a.b)if(a.g){var c=a.g;if(c.f){for(var d=0,e=null,f=null,g=c.f;g&&(g.h||(d++,g.b==a&&(e=g),!(e&&1<d)));g=g.next)e||(f=g);e&&(0==c.b&&1==d?gd(c,b):(f?(d=f,d.next==c.h&&(c.h=d),d.next=d.next.next):hd(c),id(c,e,3,b)))}a.g=null}else Zc(a,3,b)}
function jd(a,b){a.f||2!=a.b&&3!=a.b||kd(a);a.h?a.h.next=b:a.f=b;a.h=b}
function dd(a,b,c,d){var e=bd(null,null,null);e.b=new L(function(a,g){e.g=b?function(c){try{var e=b.call(d,c);a(e)}catch(m){g(m)}}:a;
e.f=c?function(b){try{var e=c.call(d,b);!q(e)&&b instanceof fd?g(b):a(e)}catch(m){g(m)}}:g});
e.b.g=a;jd(a,e);return e.b}
L.prototype.w=function(a){this.b=0;Zc(this,2,a)};
L.prototype.A=function(a){this.b=0;Zc(this,3,a)};
function Zc(a,b,c){if(0==a.b){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.b=1;a:{var d=c,e=a.w,f=a.A;if(d instanceof L){jd(d,bd(e||la,f||null,a));var g=!0}else if(Yc(d))d.then(e,f,a),g=!0;else{if(ra(d))try{var h=d.then;if(qa(h)){ld(d,h,e,f,a);g=!0;break a}}catch(l){f.call(a,l);g=!0;break a}g=!1}}g||(a.m=c,a.b=b,a.g=null,kd(a),3!=b||c instanceof fd||md(a,c))}}
function ld(a,b,c,d,e){function f(a){h||(h=!0,d.call(e,a))}
function g(a){h||(h=!0,c.call(e,a))}
var h=!1;try{b.call(a,g,f)}catch(l){f(l)}}
function kd(a){a.j||(a.j=!0,Bc(a.u,a))}
function hd(a){var b=null;a.f&&(b=a.f,a.f=b.next,b.next=null);a.f||(a.h=null);return b}
L.prototype.u=function(){for(var a;a=hd(this);)id(this,a,this.b,this.m);this.j=!1};
function id(a,b,c,d){if(3==c&&b.f&&!b.h)for(;a&&a.i;a=a.g)a.i=!1;if(b.b)b.b.g=null,nd(b,c,d);else try{b.h?b.g.call(b.context):nd(b,c,d)}catch(e){od.call(null,e)}uc(ad,b)}
function nd(a,b,c){2==b?a.g.call(a.context,c):a.f&&a.f.call(a.context,c)}
function md(a,b){a.i=!0;Bc(function(){a.i&&od.call(null,b)})}
var od=vc;function fd(a){B.call(this,a)}
z(fd,B);fd.prototype.name="cancel";function M(a){K.call(this);this.j=1;this.h=[];this.i=0;this.b=[];this.g={};this.m=!!a}
z(M,K);k=M.prototype;k.subscribe=function(a,b,c){var d=this.g[a];d||(d=this.g[a]=[]);var e=this.j;this.b[e]=a;this.b[e+1]=b;this.b[e+2]=c;this.j=e+3;d.push(e);return e};
function pd(a,b,c,d){if(b=a.g[b]){var e=a.b;(b=Da(b,function(a){return e[a+1]==c&&e[a+2]==d}))&&a.D(b)}}
k.D=function(a){var b=this.b[a];if(b){var c=this.g[b];0!=this.i?(this.h.push(a),this.b[a+1]=la):(c&&Ea(c,a),delete this.b[a],delete this.b[a+1],delete this.b[a+2])}return!!b};
k.O=function(a){var b=this.g[a];if(b){for(var c=Array(arguments.length-1),d=1,e=arguments.length;d<e;d++)c[d-1]=arguments[d];if(this.m)for(d=0;d<b.length;d++){var f=b[d];qd(this.b[f+1],this.b[f+2],c)}else{this.i++;try{for(d=0,e=b.length;d<e;d++)f=b[d],this.b[f+1].apply(this.b[f+2],c)}finally{if(this.i--,0<this.h.length&&0==this.i)for(;f=this.h.pop();)this.D(f)}}return 0!=d}return!1};
function qd(a,b,c){Bc(function(){a.apply(b,c)})}
k.clear=function(a){if(a){var b=this.g[a];b&&(C(b,this.D,this),delete this.g[a])}else this.b.length=0,this.g={}};
k.l=function(){M.o.l.call(this);this.clear();this.h.length=0};function rd(a){this.b=a}
rd.prototype.set=function(a,b){q(b)?this.b.set(a,Rc(b)):this.b.remove(a)};
rd.prototype.get=function(a){try{var b=this.b.get(a)}catch(c){return}if(null!==b)try{return Qc(b)}catch(c){throw"Storage: Invalid value was encountered";}};
rd.prototype.remove=function(a){this.b.remove(a)};function sd(a){this.b=a}
z(sd,rd);function td(a){this.data=a}
function ud(a){return!q(a)||a instanceof td?a:new td(a)}
sd.prototype.set=function(a,b){sd.o.set.call(this,a,ud(b))};
sd.prototype.f=function(a){a=sd.o.get.call(this,a);if(!q(a)||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
sd.prototype.get=function(a){if(a=this.f(a)){if(a=a.data,!q(a))throw"Storage: Invalid value was encountered";}else a=void 0;return a};function N(a){this.b=a}
z(N,sd);N.prototype.set=function(a,b,c){if(b=ud(b)){if(c){if(c<y()){N.prototype.remove.call(this,a);return}b.expiration=c}b.creation=y()}N.o.set.call(this,a,b)};
N.prototype.f=function(a,b){var c=N.o.f.call(this,a);if(c){var d;if(d=!b){d=c.creation;var e=c.expiration;d=!!e&&e<y()||!!d&&d>y()}if(d)N.prototype.remove.call(this,a);else return c}};function vd(a){this.b=a}
z(vd,N);function wd(){}
;function xd(){}
z(xd,wd);xd.prototype.clear=function(){var a=Oc(this.Y(!0)),b=this;C(a,function(a){b.remove(a)})};function yd(a){this.b=a}
z(yd,xd);k=yd.prototype;k.isAvailable=function(){if(!this.b)return!1;try{return this.b.setItem("__sak","1"),this.b.removeItem("__sak"),!0}catch(a){return!1}};
k.set=function(a,b){try{this.b.setItem(a,b)}catch(c){if(0==this.b.length)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
k.get=function(a){a=this.b.getItem(a);if(!r(a)&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
k.remove=function(a){this.b.removeItem(a)};
k.Y=function(a){var b=0,c=this.b,d=new Lc;d.next=function(){if(b>=c.length)throw Kc;var d=c.key(b++);if(a)return d;d=c.getItem(d);if(!r(d))throw"Storage mechanism: Invalid value was encountered";return d};
return d};
k.clear=function(){this.b.clear()};
k.key=function(a){return this.b.key(a)};function zd(){var a=null;try{a=window.localStorage||null}catch(b){}this.b=a}
z(zd,yd);function Ad(){var a=null;try{a=window.sessionStorage||null}catch(b){}this.b=a}
z(Ad,yd);var Bd="Microsoft Internet Explorer"==navigator.appName,O=window.performance&&window.performance.timing&&window.performance.now?function(){return window.performance.timing.navigationStart+window.performance.now()}:function(){return(new Date).getTime()};
function Cd(a,b){if(1<b.length)a[b[0]]=b[1];else{var c=b[0],d;for(d in c)a[d]=c[d]}}
;var Dd=window.yt&&window.yt.config_||window.ytcfg&&window.ytcfg.data_||{};t("yt.config_",Dd,void 0);function P(a){Cd(Dd,arguments)}
function Q(a,b){return a in Dd?Dd[a]:b}
function R(a){return Q(a,void 0)}
;function S(a,b){var c=u("yt.logging.errors.log");c?c(a,b,void 0,void 0,void 0):(c=Q("ERRORS",[]),c.push([a,b,void 0,void 0,void 0]),P("ERRORS",c))}
function Ed(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){S(b)}}:a}
;function T(a){return Q("EXPERIMENT_FLAGS",{})[a]}
;var Fd={};function Gd(a){return Fd[a]||(Fd[a]=String(a).replace(/\-([a-z])/g,function(a,c){return c.toUpperCase()}))}
function Jd(a,b){return a?a.dataset?a.dataset[Gd(b)]:a.getAttribute("data-"+b):null}
function Kd(a){a&&(a.dataset?a.dataset[Gd("loaded")]="true":a.setAttribute("data-loaded","true"))}
;function U(a,b){qa(a)&&(a=Ed(a));return window.setTimeout(a,b)}
;var Ld=u("ytPubsubPubsubInstance")||new M;M.prototype.subscribe=M.prototype.subscribe;M.prototype.unsubscribeByKey=M.prototype.D;M.prototype.publish=M.prototype.O;M.prototype.clear=M.prototype.clear;t("ytPubsubPubsubInstance",Ld,void 0);var Md=u("ytPubsubPubsubSubscribedKeys")||{};t("ytPubsubPubsubSubscribedKeys",Md,void 0);var Nd=u("ytPubsubPubsubTopicToKeys")||{};t("ytPubsubPubsubTopicToKeys",Nd,void 0);var Od=u("ytPubsubPubsubIsSynchronous")||{};t("ytPubsubPubsubIsSynchronous",Od,void 0);
function Pd(a,b){var c=Qd();if(c){var d=c.subscribe(a,function(){var c=arguments;var f=function(){Md[d]&&b.apply(window,c)};
try{Od[a]?f():U(f,0)}catch(g){S(g)}},void 0);
Md[d]=!0;Nd[a]||(Nd[a]=[]);Nd[a].push(d);return d}return 0}
function Qd(){return u("ytPubsubPubsubInstance")}
function Rd(a){Nd[a]&&(a=Nd[a],C(a,function(a){Md[a]&&delete Md[a]}),a.length=0)}
function Sd(a){var b=Qd();if(b)if(b.clear(a),a)Rd(a);else for(var c in Nd)Rd(c)}
function Td(a,b){var c=Qd();c&&c.publish.apply(c,arguments)}
function Ud(a){var b=Qd();b&&("number"==typeof a?a=[a]:r(a)&&(a=[parseInt(a,10)]),C(a,function(a){b.unsubscribeByKey(a);delete Md[a]}))}
;var Vd=/\.vflset|-vfl[a-zA-Z0-9_+=-]+/,Wd=/-[a-zA-Z]{2,3}_[a-zA-Z]{2,3}(?=(\/|$))/;function Xd(a,b){var c=Yd(a),d=document.getElementById(c),e=d&&Jd(d,"loaded"),f=d&&!e;if(e)b&&b();else{if(b){var e=Pd(c,b),g=""+(b[sa]||(b[sa]=++ta));Zd[g]=e}f||(d=$d(a,c,function(){Jd(d,"loaded")||(Kd(d),Td(c),U(w(Sd,c),0))}))}}
function $d(a,b,c){var d=document.createElement("SCRIPT");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
d.onreadystatechange=function(){switch(d.readyState){case "loaded":case "complete":d.onload()}};
Rb(d,Wb(a));a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(d,a.firstChild);return d}
function ae(a){a=Yd(a);var b=document.getElementById(a);b&&(Sd(a),b.parentNode.removeChild(b))}
function be(a,b){if(a&&b){var c=""+(b[sa]||(b[sa]=++ta));(c=Zd[c])&&Ud(c)}}
function Yd(a){var b=document.createElement("a");Qb(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"js-"+Aa(a)}
var Zd={};function ce(a,b){if(window.spf){var c="";if(a){var d=a.indexOf("jsbin/"),e=a.lastIndexOf(".js"),f=d+6;-1<d&&-1<e&&e>f&&(c=a.substring(f,e),c=c.replace(Vd,""),c=c.replace(Wd,""),c=c.replace("debug-",""),c=c.replace("tracing-",""))}spf.script.load(a,c,b)}else Xd(a,b)}
;var de=null;function ee(){var a=Q("BG_I",null),b=Q("BG_IU",null),c=Q("BG_P",void 0);b?ce(b,function(){window.botguard?fe(c):(ae(b),S(Error("Unable to load Botguard from "+b),"WARNING"))}):a&&(eval(a),fe(c))}
function fe(a){de=new window.botguard.bg(a);T("botguard_periodic_refresh")?O():T("botguard_always_refresh")}
function ge(){return null!=de}
function he(){return de?de.invoke():null}
;y();var ie=q(XMLHttpRequest)?function(){return new XMLHttpRequest}:q(ActiveXObject)?function(){return new ActiveXObject("Microsoft.XMLHTTP")}:null;
function je(){if(!ie)return null;var a=ie();return"open"in a?a:null}
function ke(a){switch(a&&"status"in a?a.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:return!0;default:return!1}}
;function le(a){"?"==a.charAt(0)&&(a=a.substr(1));a=a.split("&");for(var b={},c=0,d=a.length;c<d;c++){var e=a[c].split("=");if(1==e.length&&e[0]||2==e.length){var f=decodeURIComponent((e[0]||"").replace(/\+/g," ")),e=decodeURIComponent((e[1]||"").replace(/\+/g," "));f in b?oa(b[f])?Ga(b[f],e):b[f]=[b[f],e]:b[f]=e}}return b}
function me(a,b){var c=a.split("#",2);a=c[0];var c=1<c.length?"#"+c[1]:"",d=a.split("?",2);a=d[0];var d=le(d[1]||""),e;for(e in b)d[e]=b[e];return cc(a,d)+c}
;var ne={"X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL","X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-YouTube-Variants-Checksum":"VARIANTS_CHECKSUM"},oe=!1;
function pe(a,b){b=void 0===b?{}:b;var c=void 0;c=window.location.href;var d=I(a)[1]||null,e=$b(I(a)[3]||null);d&&e?(d=c,c=I(a),d=I(d),c=c[3]==d[3]&&c[1]==d[1]&&c[4]==d[4]):c=e?$b(I(c)[3]||null)==e&&(Number(I(c)[4]||null)||null)==(Number(I(a)[4]||null)||null):!0;for(var f in ne){if((e=d=Q(ne[f]))&&!(e=c)){var g=a,e=f,h=Q("CORS_HEADER_WHITELIST")||{};e=(g=$b(I(g)[3]||null))?(h=h[g])?0<=Ba(h,e):!1:!0}e&&(b[f]=d)}return b}
function qe(a,b){var c=Q("XSRF_FIELD_NAME",void 0),d;b.headers&&(d=b.headers["Content-Type"]);return!b.ib&&(!$b(I(a)[3]||null)||b.withCredentials||$b(I(a)[3]||null)==document.location.hostname)&&"POST"==b.method&&(!d||"application/x-www-form-urlencoded"==d)&&!(b.B&&b.B[c])}
function re(a,b){var c=b.format||"JSON";b.Da&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var d=Q("XSRF_FIELD_NAME",void 0),e=Q("XSRF_TOKEN",void 0),f=b.ab;f&&(f[d]&&delete f[d],a=me(a,f||{}));var f=b.postBody||"",g=b.B;qe(a,b)&&(g||(g={}),g[d]=e);g&&r(f)&&(d=le(f),Qa(d,g),f=b.sa&&"JSON"==b.sa?JSON.stringify(d):bc(d));d=f||g&&!Ma(g);!oe&&d&&"POST"!=b.method&&(oe=!0,S(Error("AJAX request with postData should use POST")));var h=
!1,l,m=se(a,function(a){if(!h){h=!0;l&&window.clearTimeout(l);var d=ke(a),e=null;if(d||400<=a.status&&500>a.status)e=te(c,a,b.hb);if(d)a:if(204==a.status)d=!0;else{switch(c){case "XML":d=0==parseInt(e&&e.return_code,10);break a;case "RAW":d=!0;break a}d=!!e}var e=e||{},f=b.context||p;d?b.P&&b.P.call(f,a,e):b.onError&&b.onError.call(f,a,e);b.Ia&&b.Ia.call(f,a,e)}},b.method,f,b.headers,b.responseType,b.withCredentials);
b.aa&&0<b.timeout&&(l=U(function(){h||(h=!0,m.abort(),window.clearTimeout(l),b.aa.call(b.context||p,m))},b.timeout))}
function te(a,b,c){var d=null;switch(a){case "JSON":a=b.responseText;b=b.getResponseHeader("Content-Type")||"";a&&0<=b.indexOf("json")&&(d=JSON.parse(a));break;case "XML":if(b=(b=b.responseXML)?ue(b):null)d={},C(b.getElementsByTagName("*"),function(a){d[a.tagName]=ve(a)})}c&&we(d);
return d}
function we(a){if(ra(a))for(var b in a){var c;(c="html_content"==b)||(c=b.length-5,c=0<=c&&b.indexOf("_html",c)==c);if(c){c=b;var d=Pb(a[b]);a[c]=d}else we(a[b])}}
function ue(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&0<a.length?a[0]:null:null}
function ve(a){var b="";C(a.childNodes,function(a){b+=a.nodeValue});
return b}
function xe(a,b){b.method="POST";b.B||(b.B={});re(a,b)}
function se(a,b,c,d,e,f,g){function h(){4==(l&&"readyState"in l?l.readyState:0)&&b&&Ed(b)(l)}
c=void 0===c?"GET":c;d=void 0===d?"":d;var l=je();if(!l)return null;"onloadend"in l?l.addEventListener("loadend",h,!1):l.onreadystatechange=h;l.open(c,a,!0);f&&(l.responseType=f);g&&(l.withCredentials=!0);c="POST"==c;if(e=pe(a,e))for(var m in e)l.setRequestHeader(m,e[m]),"content-type"==m.toLowerCase()&&(c=!1);c&&l.setRequestHeader("Content-Type","application/x-www-form-urlencoded");l.send(d);return l}
;var ye={},ze=0;function Ae(a,b){a&&(Q("USE_NET_AJAX_FOR_PING_TRANSPORT",!1)?se(a,b):Be(a,b))}
function Be(a,b){var c=new Image,d=""+ze++;ye[d]=c;c.onload=c.onerror=function(){b&&ye[d]&&b();delete ye[d]};
c.src=a}
;function Ce(a,b,c,d,e){b=void 0===b?"ERROR":b;e=void 0===e?!1:e;c={name:c||Q("INNERTUBE_CONTEXT_CLIENT_NAME",1),version:d||Q("INNERTUBE_CONTEXT_CLIENT_VERSION",void 0)};b=void 0===b?"ERROR":b;e=window&&window.yterr||(void 0===e?!1:e)||!1;if(a&&e&&!(5<=De)){e=a.stacktrace;d=a.columnNumber;var f=u("window.location.href");if(r(a))a={message:a,name:"Unknown error",lineNumber:"Not available",fileName:f,stack:"Not available"};else{var g=!1;try{var h=a.lineNumber||a.line||"Not available"}catch(D){h="Not available",
g=!0}try{var l=a.fileName||a.filename||a.sourceURL||p.$googDebugFname||f}catch(D){l="Not available",g=!0}a=!g&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name?a:{message:a.message||"Not available",name:a.name||"UnknownError",lineNumber:h,fileName:l,stack:a.stack||"Not available"}}e=e||a.stack;h=a.lineNumber.toString();isNaN(h)||isNaN(d)||(h=h+":"+d);if(!(Ee[a.message]||0<=e.indexOf("/YouTubeCenter.js")||0<=e.indexOf("/mytube.js"))){l=a.fileName;h={ab:{a:"logerror",t:"jserror",type:a.name,msg:a.message.substr(0,
1E3),line:h,level:b,"client.name":c.name},B:{url:Q("PAGE_NAME",window.location.href),file:l},method:"POST"};c.version&&(h["client.version"]=c.version);e&&(h.B.stack=e);for(var m in c)h.B["client."+m]=c[m];if(m=Q("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS",void 0))for(var x in m)h.B[x]=m[x];re("/error_204",h);Ee[a.message]=!0;De++}}}
var De=0,Ee={};var Fe=window.yt&&window.yt.msgs_||window.ytcfg&&window.ytcfg.msgs||{};t("yt.msgs_",Fe,void 0);function Ge(a){Cd(Fe,arguments)}
;function He(a,b){var c=5E3;isNaN(c)&&(c=void 0);var d=u("yt.scheduler.instance.addJob");return d?d(a,b,c):void 0===c?(a(),NaN):U(a,c||0)}
function Ie(a){if(!isNaN(a)){var b=u("yt.scheduler.instance.cancelJob");b?b(a):window.clearTimeout(a)}}
;var Je=[],Ke=!1;function Le(){function a(){Ke=!0;"google_ad_status"in window?P("DCLKSTAT",1):P("DCLKSTAT",2)}
ce("//static.doubleclick.net/instream/ad_status.js",a);Je.push(He(function(){Ke||"google_ad_status"in window||(be("//static.doubleclick.net/instream/ad_status.js",a),P("DCLKSTAT",3))},1))}
function Me(){return parseInt(Q("DCLKSTAT",0),10)}
;var Ne=0,Oe=u("ytDomDomGetNextId")||function(){return++Ne};
t("ytDomDomGetNextId",Oe,void 0);var Pe={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function Qe(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.shiftKey=this.ctrlKey=this.altKey=!1;this.clientY=this.clientX=0;this.changedTouches=this.touches=null;if(a=a||window.event){this.event=a;for(var b in a)b in Pe||(this[b]=a[b]);(b=a.target||a.srcElement)&&3==b.nodeType&&(b=b.parentNode);this.target=b;if(b=a.relatedTarget)try{b=b.nodeName?b:null}catch(c){b=null}else"mouseover"==this.type?b=a.fromElement:
"mouseout"==this.type&&(b=a.toElement);this.relatedTarget=b;this.clientX=void 0!=a.clientX?a.clientX:a.pageX;this.clientY=void 0!=a.clientY?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||("keypress"==this.type?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey}}
Qe.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
Qe.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
Qe.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var La=u("ytEventsEventsListeners")||{};t("ytEventsEventsListeners",La,void 0);var Re=u("ytEventsEventsCounter")||{count:0};t("ytEventsEventsCounter",Re,void 0);function Se(a,b,c,d){d=void 0===d?!1:d;a.addEventListener&&("mouseenter"!=b||"onmouseenter"in document?"mouseleave"!=b||"onmouseenter"in document?"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return Ka(function(e){return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&e[4]==!!d})}
function Te(a,b,c){var d=void 0===d?!1:d;if(!a||!a.addEventListener&&!a.attachEvent)return"";var e=Se(a,b,c,d);if(e)return e;var e=++Re.count+"",f=!("mouseenter"!=b&&"mouseleave"!=b||!a.addEventListener||"onmouseenter"in document);var g=f?function(d){d=new Qe(d);if(!Vb(d.relatedTarget,function(b){return b==a}))return d.currentTarget=a,d.type=b,c.call(a,d)}:function(b){b=new Qe(b);
b.currentTarget=a;return c.call(a,b)};
g=Ed(g);a.addEventListener?("mouseenter"==b&&f?b="mouseover":"mouseleave"==b&&f?b="mouseout":"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),a.addEventListener(b,g,d)):a.attachEvent("on"+b,g);La[e]=[a,b,c,g,d];return e}
function Ue(a){a&&("string"==typeof a&&(a=[a]),C(a,function(a){if(a in La){var b=La[a],d=b[0],e=b[1],f=b[3],b=b[4];d.removeEventListener?d.removeEventListener(e,f,b):d.detachEvent&&d.detachEvent("on"+e,f);delete La[a]}}))}
;function Ve(){if(null==u("_lact",window)){var a=parseInt(Q("LACT"),10),a=isFinite(a)?y()-Math.max(a,0):-1;t("_lact",a,window);t("_fact",a,window);-1==a&&V();Te(document,"keydown",V);Te(document,"keyup",V);Te(document,"mousedown",V);Te(document,"mouseup",V);Pd("page-mouse",V);Pd("page-scroll",V);Pd("page-resize",V)}}
function V(){null==u("_lact",window)&&(Ve(),u("_lact",window));var a=y();t("_lact",a,window);-1==u("_fact",window)&&t("_fact",a,window);Td("USER_ACTIVE")}
function We(){var a=u("_lact",window);return null==a?-1:Math.max(y()-a,0)}
var Xe=V;function Ye(a,b,c,d,e){this.g=a;this.i=b;this.h=c;this.f=d;this.b=e}
var Ze=1;function $e(a){var b={};void 0!==a.g?b.trackingParams=a.g:(b.veType=a.i,null!=a.h&&(b.veCounter=a.h),null!=a.f&&(b.elementIndex=a.f));void 0!==a.b&&(b.dataElement=$e(a.b));return b}
;var af={log_event:"events",log_event2:"events",log_interaction:"interactions"},bf=Object.create(null);bf.log_event="GENERIC_EVENT_LOGGING";bf.log_event2="GENERIC_EVENT_LOGGING";bf.log_interaction="INTERACTION_LOGGING";var cf={},df={},ef=0,W=u("ytLoggingTransportLogPayloadsQueue_")||{};t("ytLoggingTransportLogPayloadsQueue_",W,void 0);var ff=u("ytLoggingTransportTokensToCttTargetIds_")||{};t("ytLoggingTransportTokensToCttTargetIds_",ff,void 0);var gf=u("ytLoggingTransportDispatchedStats_")||{};
t("ytLoggingTransportDispatchedStats_",gf,void 0);var hf=u("ytLoggingTransportCapturedTime_")||{};t("ytytLoggingTransportCapturedTime_",hf,void 0);function jf(a,b){df[a.endpoint]=b;if(a.Z){var c=a.Z;var d={};c.videoId?d.videoId=c.videoId:c.playlistId&&(d.playlistId=c.playlistId);ff[a.Z.token]=d;c=kf(a.endpoint,a.Z.token)}else c=kf(a.endpoint);c.push(a.ra);d=T("web_logging_max_batch");c.length>=(Number(d||0)||20)?lf():mf()}
function lf(){window.clearTimeout(ef);if(!Ma(W)){for(var a in W){var b=cf[a];if(!b){var c=df[a];if(!c)continue;b=new c;cf[a]=b}var c=void 0,d=a,e=b,f=af[d],g=gf[d]||{};gf[d]=g;b=Math.round(O());for(c in W[d]){var h=e.b,h={client:{hl:h.Ga,gl:h.Fa,clientName:h.Ea,clientVersion:h.innertubeContextClientVersion}};Q("DELEGATED_SESSION_ID")&&(h.user={onBehalfOfUser:Q("DELEGATED_SESSION_ID")});h={context:h};h[f]=kf(d,c);g.dispatchedEventCount=g.dispatchedEventCount||0;g.dispatchedEventCount+=h[f].length;
h.requestTimeMs=b;var l=ff[c];if(l)a:{var m=h,x=c;if(l.videoId)var D="VIDEO";else if(l.playlistId)D="PLAYLIST";else break a;m.credentialTransferTokenTargetId=l;m.context=m.context||{};m.context.user=m.context.user||{};m.context.user.credentialTransferTokens=[{token:x,scope:D}]}delete ff[c];nf(e,d,h)}c=g;d=b;c.previousDispatchMs&&(b=d-c.previousDispatchMs,e=c.diffCount||0,c.averageTimeBetweenDispatchesMs=e?(c.averageTimeBetweenDispatchesMs*e+b)/(e+1):b,c.diffCount=e+1);c.previousDispatchMs=d;delete W[a]}Ma(W)||
mf()}}
function mf(){window.clearTimeout(ef);ef=U(lf,Q("LOGGING_BATCH_TIMEOUT",1E4))}
function kf(a,b){b||(b="");W[a]=W[a]||{};W[a][b]=W[a][b]||[];return W[a][b]}
;function of(a,b,c,d,e){var f={};f.eventTimeMs=Math.round(d||O());f[a]=b;f.context={lastActivityMs:String(d?-1:We())};a=T("web_system_health_gel2")&&"systemHealthCaptured"==a?"log_event2":"log_event";jf({endpoint:a,ra:f,Z:e},c)}
;function pf(a,b,c){qf(rf,{attachChild:{csn:a,parentVisualElement:$e(b),visualElements:[$e(c)]}},void 0)}
function sf(a,b){var c=rf;if(T("interaction_logging_on_gel_web"))b.forEach(function(b){of("visualElementShown",{csn:a,ve:$e(b),eventType:1},c)});
else{var d=Ca(b,function(a){return $e(a)});
qf(c,{visibilityUpdate:{csn:a,visualElements:d}})}}
function qf(a,b,c,d){b.eventTimeMs=Math.round(O());b.lactMs=We();d?b.clientData=d:c&&(b.clientData=tf(c));jf({endpoint:"log_interaction",ra:b},a)}
function tf(a){var b={};a.analyticsChannelData&&(b.analyticsDatas=Ca(a.analyticsChannelData,function(a){return{tabName:a.tabName,cardName:a.cardName,isChannelScreen:a.isChannelScreen,insightId:a.insightId,externalChannelId:a.externalChannelId,externalContentOwnerId:a.externalContentOwnerId}}));
return{playbackData:{clientPlaybackNonce:a.clientPlaybackNonce},analyticsChannelData:b,externalLinkData:a.externalLinkData}}
;function uf(){if(!vf&&!wf||!window.JSON)return null;try{var a=vf.get("yt-player-two-stage-token")}catch(b){}if(!r(a))try{a=wf.get("yt-player-two-stage-token")}catch(b){}if(!r(a))return null;try{a=JSON.parse(a,void 0)}catch(b){}return a}
var wf,xf=new zd;wf=xf.isAvailable()?new vd(xf):null;var vf,yf=new Ad;vf=yf.isAvailable()?new vd(yf):null;function zf(){var a=Q("ROOT_VE_TYPE",void 0);return a?new Ye(void 0,a,void 0):null}
function Af(){var a=Q("client-screen-nonce",void 0);a||(a=Q("EVENT_ID",void 0));return a}
;function Bf(a,b,c){rc.set(""+a,b,c,"/","youtube.com")}
;function Cf(a){if(a){a=a.itct||a.ved;var b=u("yt.logging.screen.storeParentElement");a&&b&&b(new Ye(a))}}
;function Df(a,b,c){b=void 0===b?{}:b;c=void 0===c?!1:c;var d=Q("EVENT_ID");d&&(b.ei||(b.ei=d));if(b){var d=a,e=Q("VALID_SESSION_TEMPDATA_DOMAINS",[]),f=$b(I(window.location.href)[3]||null);f&&e.push(f);f=$b(I(d)[3]||null);if(0<=Ba(e,f)||!f&&0==d.lastIndexOf("/",0))if(T("autoescape_tempdata_url")&&(e=document.createElement("a"),Qb(e,d),d=e.href),d){var f=I(d),d=f[5],e=f[6],f=f[7],g="";d&&(g+=d);e&&(g+="?"+e);f&&(g+="#"+f);d=g;e=d.indexOf("#");if(d=0>e?d:d.substr(0,e)){if(b.itct||b.ved)b.csn=b.csn||
Af();if(h){var h=parseInt(h,10);isFinite(h)&&0<h&&(d="ST-"+Aa(d).toString(36),e=b?bc(b):"",Bf(d,e,h||5),Cf(b))}else h="ST-"+Aa(d).toString(36),d=b?bc(b):"",Bf(h,d,5),Cf(b)}}}if(c)return!1;if((window.ytspf||{}).enabled)spf.navigate(a);else{var l=void 0===l?{}:l;var m=void 0===m?"":m;var x=void 0===x?window:x;c=x.location;a=cc(a,l)+m;a=a instanceof H?a:Mb(a);c.href=Kb(a)}return!0}
;var Ef=u("yt.abuse.botguardInitialized")||ge;t("yt.abuse.botguardInitialized",Ef,void 0);var Ff=u("yt.abuse.invokeBotguard")||he;t("yt.abuse.invokeBotguard",Ff,void 0);var Gf=u("yt.abuse.dclkstatus.checkDclkStatus")||Me;t("yt.abuse.dclkstatus.checkDclkStatus",Gf,void 0);var Hf=u("yt.player.exports.navigate")||Df;t("yt.player.exports.navigate",Hf,void 0);var If=u("yt.util.activity.init")||Ve;t("yt.util.activity.init",If,void 0);var Jf=u("yt.util.activity.getTimeSinceActive")||We;
t("yt.util.activity.getTimeSinceActive",Jf,void 0);var Kf=u("yt.util.activity.setTimestamp")||Xe;t("yt.util.activity.setTimestamp",Kf,void 0);function Lf(a,b,c){c.context&&c.context.capabilities&&(c=c.context.capabilities,c.snapshot||c.golden)&&(a="vix");return"/youtubei/"+a+"/"+b}
;function rf(a){this.b=a||{apiaryHost:R("APIARY_HOST"),fb:R("APIARY_HOST_FIRSTPARTY"),gapiHintOverride:!!Q("GAPI_HINT_OVERRIDE",void 0),gapiHintParams:R("GAPI_HINT_PARAMS"),innertubeApiKey:R("INNERTUBE_API_KEY"),innertubeApiVersion:R("INNERTUBE_API_VERSION"),Ea:Q("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),innertubeContextClientVersion:R("INNERTUBE_CONTEXT_CLIENT_VERSION"),Ga:R("INNERTUBE_CONTEXT_HL"),Fa:R("INNERTUBE_CONTEXT_GL"),xhrApiaryHost:R("XHR_APIARY_HOST")||"",Ha:R("INNERTUBE_HOST_OVERRIDE")||""}}
function nf(a,b,c){var d={};!Q("VISITOR_DATA")&&.01>Math.random()&&S(Error("Missing VISITOR_DATA when sending innertube request."),"WARNING");var e={headers:{"Content-Type":"application/json","X-Goog-Visitor-Id":Q("VISITOR_DATA","")},B:c,sa:"JSON",aa:d.aa,P:function(a,b){d.P&&d.P(b)},
onError:function(a,b){if(d.onError)d.onError(b)},
timeout:d.timeout,withCredentials:!0},f=sc();f&&(e.headers.Authorization=f,e.headers["X-Goog-AuthUser"]=Q("SESSION_INDEX",0));var g="",h=a.b.Ha;h&&(g=h);f&&!g&&(e.headers["x-origin"]=window.location.origin);xe(""+g+Lf(a.b.innertubeApiVersion,b,c)+"?alt=json&key="+a.b.innertubeApiKey,e)}
;function Mf(a){a=a||{};this.url=a.url||"";this.urlV9As2=a.url_v9as2||"";this.args=a.args||Oa(Nf);this.assets=a.assets||{};this.attrs=a.attrs||Oa(Of);this.params=a.params||Oa(Pf);this.minVersion=a.min_version||"8.0.0";this.fallback=a.fallback||null;this.fallbackMessage=a.fallbackMessage||null;this.html5=!!a.html5;this.disable=a.disable||{};this.loaded=!!a.loaded;this.messages=a.messages||{}}
var Nf={enablejsapi:1},Of={},Pf={allowscriptaccess:"always",allowfullscreen:"true",bgcolor:"#000000"};function Qf(a){a instanceof Mf||(a=new Mf(a));return a}
function Rf(a){var b=new Mf,c;for(c in a)if(a.hasOwnProperty(c)){var d=a[c];b[c]="object"==na(d)?Oa(d):d}return b}
;function Sf(a){K.call(this);this.b=[];this.g=a||this}
n(Sf,K);function Tf(a,b,c,d){d=Ed(v(d,a.g));d={target:b,name:c,na:d};b.addEventListener(c,d.na,void 0);a.b.push(d)}
function Uf(a){for(;a.b.length;){var b=a.b.pop();b.target.removeEventListener(b.name,b.na)}}
Sf.prototype.l=function(){Uf(this);K.prototype.l.call(this)};function Vf(){this.g=this.f=this.b=0;this.h="";var a=u("window.navigator.plugins"),b=u("window.navigator.mimeTypes"),a=a&&a["Shockwave Flash"],b=b&&b["application/x-shockwave-flash"],b=a&&b&&b.enabledPlugin&&a.description||"";if(a=b){var c=a.indexOf("Shockwave Flash");0<=c&&(a=a.substr(c+15));for(var c=a.split(" "),d="",a="",e=0,f=c.length;e<f;e++)if(d)if(a)break;else a=c[e];else d=c[e];d=d.split(".");c=parseInt(d[0],10)||0;d=parseInt(d[1],10)||0;e=0;if("r"==a.charAt(0)||"d"==a.charAt(0))e=parseInt(a.substr(1),
10)||0;a=[c,d,e]}else a=[0,0,0];this.h=b;b=a;this.b=b[0];this.f=b[1];this.g=b[2];if(0>=this.b){if(Bd)try{var g=new ActiveXObject("ShockwaveFlash.ShockwaveFlash")}catch(x){g=null}else{var h=document.body;var l=document.createElement("object");l.setAttribute("type","application/x-shockwave-flash");g=h.appendChild(l)}if(g&&"GetVariable"in g)try{var m=g.GetVariable("$version")}catch(x){m=""}h&&l&&h.removeChild(l);(g=m||"")?(g=g.split(" ")[1].split(","),g=[parseInt(g[0],10)||0,parseInt(g[1],10)||0,parseInt(g[2],
10)||0]):g=[0,0,0];this.b=g[0];this.f=g[1];this.g=g[2]}}
ma(Vf);function Wf(a,b,c,d){b="string"==typeof b?b.split("."):[b,c,d];b[0]=parseInt(b[0],10)||0;b[1]=parseInt(b[1],10)||0;b[2]=parseInt(b[2],10)||0;return a.b>b[0]||a.b==b[0]&&a.f>b[1]||a.b==b[0]&&a.f==b[1]&&a.g>=b[2]}
;var Xf=/cssbin\/(?:debug-)?([a-zA-Z0-9_-]+?)(?:-2x|-web|-rtl|-vfl|.css)/;function Yf(a){a=a||"";if(window.spf){var b=a.match(Xf);spf.style.load(a,b?b[1]:"",void 0)}else Zf(a)}
function Zf(a){var b=$f(a),c=document.getElementById(b),d=c&&Jd(c,"loaded");d||c&&!d||(c=ag(a,b,function(){Jd(c,"loaded")||(Kd(c),Td(b),U(w(Sd,b),0))}))}
function ag(a,b,c){var d=document.createElement("link");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
a=Wb(a);d.rel="stylesheet";d.href=Ib(a);(document.getElementsByTagName("head")[0]||document.body).appendChild(d);return d}
function $f(a){var b=document.createElement("A");a=Nb(a);Qb(b,a);b=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"css-"+Aa(b)}
;var X={},bg=(X["api.invalidparam"]=2,X.auth=150,X["drm.auth"]=150,X["heartbeat.net"]=150,X["heartbeat.servererror"]=150,X["heartbeat.stop"]=150,X["html5.unsupportedads"]=5,X["fmt.noneavailable"]=5,X["fmt.decode"]=5,X["fmt.unplayable"]=5,X["html5.missingapi"]=5,X["html5.unsupportedlive"]=5,X["drm.unavailable"]=5,X);var cg={cupcake:1.5,donut:1.6,eclair:2,froyo:2.2,gingerbread:2.3,honeycomb:3,"ice cream sandwich":4,jellybean:4.1,kitkat:4.4,lollipop:5.1,marshmallow:6,nougat:7.1},dg;var eg=E,eg=eg.toLowerCase();if(-1!=eg.indexOf("android")){var fg=eg.match(/android\D*(\d\.\d)[^\;|\)]*[\;\)]/);if(fg)dg=parseFloat(fg[1]);else{var gg=[],hg=0,ig;for(ig in cg)gg[hg++]=ig;var jg=eg.match("("+gg.join("|")+")");dg=jg?cg[jg[0]]:0}}else dg=void 0;var kg=['video/mp4; codecs="avc1.42001E, mp4a.40.2"','video/webm; codecs="vp8.0, vorbis"'],lg=['audio/mp4; codecs="mp4a.40.2"'];var mg=u("ytLoggingLatencyUsageStats_")||{};t("ytLoggingLatencyUsageStats_",mg,void 0);var ng=0;function og(a){mg[a]=mg[a]||{count:0};var b=mg[a];b.count++;b.time=O();ng||(ng=He(pg,0));return 10<b.count?(11==b.count&&Ce(Error("CSI data exceeded logging limit with key: "+a)),!0):!1}
function pg(){var a=O(),b;for(b in mg)6E4<a-mg[b].time&&delete mg[b];ng=0}
;function qg(a,b){this.version=a;this.args=b}
;function rg(a){this.topic=a}
rg.prototype.toString=function(){return this.topic};var sg=u("ytPubsub2Pubsub2Instance")||new M;M.prototype.subscribe=M.prototype.subscribe;M.prototype.unsubscribeByKey=M.prototype.D;M.prototype.publish=M.prototype.O;M.prototype.clear=M.prototype.clear;t("ytPubsub2Pubsub2Instance",sg,void 0);var tg=u("ytPubsub2Pubsub2SubscribedKeys")||{};t("ytPubsub2Pubsub2SubscribedKeys",tg,void 0);var ug=u("ytPubsub2Pubsub2TopicToKeys")||{};t("ytPubsub2Pubsub2TopicToKeys",ug,void 0);var vg=u("ytPubsub2Pubsub2IsAsync")||{};t("ytPubsub2Pubsub2IsAsync",vg,void 0);
t("ytPubsub2Pubsub2SkipSubKey",null,void 0);function wg(a,b){var c=u("ytPubsub2Pubsub2Instance");c&&c.publish.call(c,a.toString(),a,b)}
;function xg(){var a=Q("TIMING_TICK_EXPIRATION");a||(a={},P("TIMING_TICK_EXPIRATION",a));return a}
function yg(){var a=xg(),b;for(b in a)Ie(a[b]);P("TIMING_TICK_EXPIRATION",{})}
;function zg(a,b){qg.call(this,1,arguments)}
n(zg,qg);function Ag(a,b){qg.call(this,1,arguments)}
n(Ag,qg);var Bg=new rg("aft-recorded"),Cg=new rg("timing-sent");var Y=window.performance||window.mozPerformance||window.msPerformance||window.webkitPerformance||{};var Dg=y().toString();var Eg={vc:!0},Fg={ad_at:"adType",ad_cpn:"adClientPlaybackNonce",ad_docid:"adVideoId",cpn:"clientPlaybackNonce",csn:"clientScreenNonce",docid:"videoId",is_nav:"isNavigation",yt_lt:"loadType",yt_ad:"isMonetized",plid:"videoId",fmt:"playerInfo.itag",yt_ad_pr:"prerollAllowed",yt_red:"isRedSubscriber",st:"serverTimeMs",yt_vis:"isVisible"},Gg="ap c cver ei srt yt_fss yt_li GetBrowse_rid GetPlayer_rid GetSearch_rid GetWatchNext_rid ad_allowed ad_docid ba cmt ncnp nr p pa paused pc prerender psc rc start vpil vpni vps yt_abt yt_ad_an yt_eil yt_fn yt_fs yt_pft yt_pl yt_pre yt_pt yt_pvis yt_ref yt_sts".split(" "),
Hg=["isNavigation","isMonetized","prerollAllowed","isRedSubscriber","isVisible"],Ig=!1;
function Jg(a){if("_"!=a[0]){var b=a;Y.mark&&(0==b.lastIndexOf("mark_",0)||(b="mark_"+b),Y.mark(b))}var b=Kg(),c=O();b[a]&&(b["_"+a]=b["_"+a]||[b[a]],b["_"+a].push(c));b[a]=c;b=xg();if(c=b[a])Ie(c),b[a]=0;Lg()["tick_"+a]=void 0;O();T("csi_on_gel")?(b=Mg(),"_start"==a?og("baseline_"+b)||of("latencyActionBaselined",{clientActionNonce:b},rf,void 0):og("tick_"+a+"_"+b)||of("latencyActionTicked",{tickName:a,clientActionNonce:b},rf,void 0),a=!0):a=!1;if(a=!a)a=!u("yt.timing.pingSent_");if(a&&(b=R("TIMING_ACTION"),
a=Kg(),u("ytglobal.timingready_")&&b&&a._start&&(b=Ng()))){T("tighter_critical_section")&&!Ig&&(wg(Bg,new zg(Math.round(b-a._start),void 0)),Ig=!0);b=!0;c=Q("TIMING_WAIT",[]);if(c.length)for(var d=0,e=c.length;d<e;++d)if(!(c[d]in a)){b=!1;break}b&&Og()}}
function Pg(){var a=Qg().info.yt_lt="hot_bg";Lg().info_yt_lt=a;if(T("csi_on_gel"))if("yt_lt"in Fg){var b={},c=Fg.yt_lt.split(".");0<=Ba(Hg,c)&&(a=!!a);for(var d=b,e=0;e<c.length-1;e++)d[c[e]]=d[c[e]]||{},d=d[c[e]];b[c[c.length-1]]=a;a=Mg();c=Object.keys(b).join("");og("info_"+c+"_"+a)||(b.clientActionNonce=a,of("latencyActionInfo",b,rf))}else 0<=Ba(Gg,"yt_lt")||S(Error("Unknown label yt_lt logged with GEL CSI."))}
function Ng(){var a=Kg();if(a.aft)return a.aft;for(var b=Q("TIMING_AFT_KEYS",["ol"]),c=b.length,d=0;d<c;d++){var e=a[b[d]];if(e)return e}return NaN}
function Og(){yg();if(!T("csi_on_gel")){var a=Kg(),b=Qg().info,c=a._start,d;for(d in a)if(0==d.lastIndexOf("_",0)&&oa(a[d])){var e=d.slice(1);if(e in Eg){var f=Ca(a[d],function(a){return Math.round(a-c)});
b["all_"+e]=f.join()}delete a[d]}e=!!b.ap;if(f=u("ytglobal.timingReportbuilder_")){if(f=f(a,b,void 0))Rg(f,e),Sg(),Tg(),Ug(!1,void 0),Q("TIMING_ACTION")&&P("PREVIOUS_ACTION",Q("TIMING_ACTION")),P("TIMING_ACTION","")}else{var g=Q("CSI_SERVICE_NAME","youtube");f={v:2,s:g,action:Q("TIMING_ACTION",void 0)};var h=b.srt;void 0!==a.srt&&delete b.srt;if(b.h5jse){var l=window.location.protocol+u("ytplayer.config.assets.js");(l=Y.getEntriesByName?Y.getEntriesByName(l)[0]:null)?b.h5jse=Math.round(b.h5jse-l.responseEnd):
delete b.h5jse}a.aft=Ng();Vg()&&"youtube"==g&&(Pg(),g=a.vc,l=a.pbs,delete a.aft,b.aft=Math.round(l-g));for(var m in b)"_"!=m.charAt(0)&&(f[m]=b[m]);a.ps=O();b={};m=[];for(d in a)"_"!=d.charAt(0)&&(g=Math.round(a[d]-c),b[d]=g,m.push(d+"."+g));f.rt=m.join(",");(a=u("ytdebug.logTiming"))&&a(f,b);Rg(f,e,void 0);wg(Cg,new Ag(b.aft+(h||0),void 0))}}}
var Tg=v(Y.clearResourceTimings||Y.webkitClearResourceTimings||Y.mozClearResourceTimings||Y.msClearResourceTimings||Y.oClearResourceTimings||la,Y);
function Rg(a,b,c){if(T("debug_csi_data")){var d=u("yt.timing.csiData");d||(d=[],t("yt.timing.csiData",d,void 0));d.push({page:location.href,time:new Date,args:a})}var d="",e;for(e in a)d+="&"+e+"="+a[e];a="/csi_204?"+d.substring(1);if(window.navigator&&window.navigator.sendBeacon&&b)try{window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,"")||Ae(a,void 0)}catch(f){Ae(a,void 0)}else Ae(a);Ug(!0,c)}
function Mg(){var a=Qg().nonce;if(!a){a:{if(window.crypto&&window.crypto.getRandomValues)try{var b=Array(16),c=new Uint8Array(16);window.crypto.getRandomValues(c);for(a=0;a<b.length;a++)b[a]=c[a];var d=b;break a}catch(e){}d=Array(16);for(b=0;16>b;b++){c=y();for(a=0;a<c%23;a++)d[b]=Math.random();d[b]=Math.floor(256*Math.random())}if(Dg)for(b=1,c=0;c<Dg.length;c++)d[b%16]=d[b%16]^d[(b-1)%16]/4^Dg.charCodeAt(c),b++}b=[];for(c=0;c<d.length;c++)b.push("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_".charAt(d[c]&
63));a=b.join("");Qg().nonce=a}return a}
function Kg(){return Qg().tick}
function Lg(){var a=Qg();"gel"in a||(a.gel={});return a.gel}
function Qg(){return u("ytcsi.data_")||Sg()}
function Sg(){var a={tick:{},info:{}};t("ytcsi.data_",a,void 0);return a}
function Ug(a,b){t("yt.timing."+(b||"")+"pingSent_",a,void 0)}
function Vg(){var a=Kg(),b=a.pbr,c=a.vc,a=a.pbs;return b&&c&&a&&b<c&&c<a&&1==Qg().info.yt_pvis}
;function Wg(a,b){K.call(this);this.u=this.j=a;this.M=b;this.A=!1;this.g={};this.V=this.L=null;this.N=new M;Hc(this,w(Ic,this.N));this.i={};this.G=this.W=this.h=this.ea=this.b=null;this.R=!1;this.H=this.w=this.m=this.K=null;this.X={};this.wa=["onReady"];this.T=new Sf(this);Hc(this,w(Ic,this.T));this.ca=null;this.la=NaN;this.U={};Xg(this);this.C("onDetailedError",v(this.La,this));this.C("onTabOrderChange",v(this.xa,this));this.C("onTabAnnounce",v(this.ma,this));this.C("WATCH_LATER_VIDEO_ADDED",v(this.Ma,
this));this.C("WATCH_LATER_VIDEO_REMOVED",v(this.Na,this));zb||(this.C("onMouseWheelCapture",v(this.Ja,this)),this.C("onMouseWheelRelease",v(this.Ka,this)));this.C("onAdAnnounce",v(this.ma,this));this.I=new Sf(this);Hc(this,w(Ic,this.I));this.da=!1;this.ba=null}
z(Wg,K);var Yg=["fmt.noneavailable","html5.missingapi","html5.unsupportedads","html5.unsupportedlive"];k=Wg.prototype;k.ka=function(a,b){this.f||(Zg(this,a),$g(this,b),this.A&&ah(this))};
function Zg(a,b){a.ea=b;a.b=Rf(b);a.h=a.b.attrs.id||a.h;"video-player"==a.h&&(a.h=a.M,a.b.attrs.id=a.M);a.u.id==a.h&&(a.h+="-player",a.b.attrs.id=a.h);a.b.args.enablejsapi="1";a.b.args.playerapiid=a.M;a.W||(a.W=bh(a,a.b.args.jsapicallback||"onYouTubePlayerReady"));a.b.args.jsapicallback=null;var c=a.b.attrs.width;c&&(a.u.style.width=Yb(Number(c)||c));if(c=a.b.attrs.height)a.u.style.height=Yb(Number(c)||c)}
k.Aa=function(){return this.ea};
function ah(a){a.b.loaded||(a.b.loaded=!0,"0"!=a.b.args.autoplay?a.g.loadVideoByPlayerVars(a.b.args):a.g.cueVideoByPlayerVars(a.b.args))}
function ch(a){var b=a.b&&a.b.args;if(b&&b.fflags){var c=b.el,b=b.fflags;if(("detailpage"==c||"profilepage"==c)&&0<=b.indexOf("web_player_disable_flash_watch=true")||"embedded"==c&&0<=b.indexOf("web_player_disable_flash_embed=true"))return!1}q(a.b.disable.flash)||(c=a.b.disable,b=Wf(Vf.getInstance(),a.b.minVersion),c.flash=!b);return!a.b.disable.flash}
function dh(a,b){var c=a.b,d=c&&c.args&&c.args.fflags;!d||0>d.indexOf("web_player_flash_fallback_killswitch=true")||b&&(5!=(bg[b.errorCode]||5)||-1==Yg.indexOf(b.errorCode))||!ch(a)||((d=eh(a))&&d.stopVideo&&d.stopVideo(),d&&d.getUpdatedConfigurationData&&(c=d.getUpdatedConfigurationData(),c=Qf(c)),c.args.autoplay=1,c.args.html5_unavailable="1",Zg(a,c),$g(a,"flash"))}
function $g(a,b){if(!a.f){if(!b){var c;if(!(c=!a.b.html5&&ch(a))){if(!q(a.b.disable.html5)){c=!0;void 0!=a.b.args.deviceHasDisplay&&(c=a.b.args.deviceHasDisplay);if(2.2==dg)var d=!0;else{a:{var e=c;c=u("yt.player.utils.videoElement_");c||(c=document.createElement("VIDEO"),t("yt.player.utils.videoElement_",c,void 0));try{if(c.canPlayType)for(var e=e?kg:lg,f=0;f<e.length;f++)if(c.canPlayType(e[f])){d=null;break a}d="fmt.noneavailable"}catch(g){d="html5.missingapi"}}d=!d}d&&(d=fh(a)||a.b.assets.js);
a.b.disable.html5=!d;d||(a.b.args.html5_unavailable="1")}c=!!a.b.disable.html5}b=c?ch(a)?"flash":"unsupported":"html5"}("flash"==b?a.bb:a.cb).call(a)}}
function fh(a){var b=!0,c=eh(a);c&&a.b&&(a=a.b,b=Jd(c,"version")==a.assets.js);return b&&!!u("yt.player.Application.create")}
k.cb=function(){if(!this.R){var a=fh(this);if(a&&"html5"==gh(this))this.G="html5",this.A||this.J();else if(hh(this),this.G="html5",a&&this.m)this.j.appendChild(this.m),this.J();else{this.b.loaded=!0;var b=!1;this.K=v(function(){b=!0;var a=this.j,d=Rf(this.b);u("yt.player.Application.create")(a,d);this.J()},this);
this.R=!0;a?this.K():(ce(this.b.assets.js,this.K),Yf(this.b.assets.css),ih(this)&&!b&&t("yt.player.Application.create",null,void 0))}}};
k.bb=function(){var a=Rf(this.b);if(!this.w){var b=eh(this);b&&(this.w=document.createElement("SPAN"),this.w.tabIndex=0,Tf(this.T,this.w,"focus",this.pa),this.H=document.createElement("SPAN"),this.H.tabIndex=0,Tf(this.T,this.H,"focus",this.pa),b.parentNode&&b.parentNode.insertBefore(this.w,b),b.parentNode&&b.parentNode.insertBefore(this.H,b.nextSibling))}a.attrs.width=a.attrs.width||"100%";a.attrs.height=a.attrs.height||"100%";if("flash"==gh(this))this.G="flash",this.A||this.J();else{hh(this);this.G=
"flash";this.b.loaded=!0;var b=Vf.getInstance(),c=(-1<b.h.indexOf("Gnash")&&-1==b.h.indexOf("AVM2")||9==b.b&&1==b.f||9==b.b&&0==b.f&&1==b.g?0:9<=b.b)||-1<navigator.userAgent.indexOf("Sony/COM2")&&!Wf(b,9,1,58)?a.url:a.urlV9As2;window!=window.top&&document.referrer&&(a.args.framer=document.referrer.substring(0,128));b=this.j;if(c){var b=r(b)?Sb(b):b,d=Oa(a.attrs);d.tabindex=0;var e=Oa(a.params);e.flashvars=bc(a.args);if(Bd){d.classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000";e.movie=c;var a=document.createElement("object");
for(g in d)a.setAttribute(g,d[g]);for(var f in e){var g=document.createElement("param");g.setAttribute("name",f);g.setAttribute("value",e[f]);a.appendChild(g)}}else{d.type="application/x-shockwave-flash";d.src=c;a=document.createElement("embed");a.setAttribute("name",d.id);for(var h in d)a.setAttribute(h,d[h]);for(var l in e)a.setAttribute(l,e[l])}f=document.createElement("div");f.appendChild(a);b.innerHTML=f.innerHTML}this.J()}};
k.pa=function(){eh(this).focus()};
function eh(a){var b=Sb(a.h);!b&&a.u&&a.u.querySelector&&(b=a.u.querySelector("#"+a.h));return b}
k.J=function(){if(!this.f){var a=eh(this),b=!1;try{a&&a.getApiInterface&&a.getApiInterface()&&(b=!0)}catch(f){}if(b)if(this.R=!1,a.isNotServable&&a.isNotServable(this.b.args.video_id))dh(this);else{Xg(this);this.A=!0;a=eh(this);a.addEventListener&&(this.L=jh(this,a,"addEventListener"));a.removeEventListener&&(this.V=jh(this,a,"removeEventListener"));for(var b=a.getApiInterface(),b=b.concat(a.getInternalApiInterface()),c=0;c<b.length;c++){var d=b[c];this.g[d]||(this.g[d]=jh(this,a,d))}for(var e in this.i)this.L(e,
this.i[e]);ah(this);this.W&&this.W(this.g);this.N.O("onReady",this.g)}else this.la=U(v(this.J,this),50)}};
function jh(a,b,c){var d=b[c];return function(){try{return a.ca=null,d.apply(b,arguments)}catch(e){"Bad NPObject as private data!"!=e.message&&"sendAbandonmentPing"!=c&&(e.message+=" ("+c+")",a.ca=e,S(e,"WARNING"))}}}
function Xg(a){a.A=!1;if(a.V)for(var b in a.i)a.V(b,a.i[b]);for(var c in a.U)window.clearTimeout(parseInt(c,10));a.U={};a.L=null;a.V=null;for(var d in a.g)a.g[d]=null;a.g.addEventListener=v(a.C,a);a.g.removeEventListener=v(a.Sa,a);a.g.destroy=v(a.dispose,a);a.g.getLastError=v(a.Ba,a);a.g.getPlayerType=v(a.Ca,a);a.g.getCurrentVideoConfig=v(a.Aa,a);a.g.loadNewVideoConfig=v(a.ka,a);a.g.isReady=v(a.eb,a)}
k.eb=function(){return this.A};
k.C=function(a,b){if(!this.f){var c=bh(this,b);if(c){if(!(0<=Ba(this.wa,a)||this.i[a])){var d=kh(this,a);this.L&&this.L(a,d)}this.N.subscribe(a,c);"onReady"==a&&this.A&&U(w(c,this.g),0)}}};
k.Sa=function(a,b){if(!this.f){var c=bh(this,b);c&&pd(this.N,a,c)}};
function bh(a,b){var c=b;if("string"==typeof b){if(a.X[b])return a.X[b];c=function(){var a=u(b);a&&a.apply(p,arguments)};
a.X[b]=c}return c?c:null}
function kh(a,b){var c="ytPlayer"+b+a.M;a.i[b]=c;p[c]=function(c){var d=U(function(){if(!a.f){a.N.O(b,c);var e=a.U,g=String(d);g in e&&delete e[g]}},0);
Na(a.U,String(d))};
return c}
k.xa=function(a){a=a?Ub:Tb;for(var b=a(document.activeElement);b&&(1!=b.nodeType||b==this.w||b==this.H||(b.focus(),b!=document.activeElement));)b=a(b)};
k.ma=function(a){Td("a11y-announce",a)};
k.La=function(a){dh(this,a)};
k.Ma=function(a){Td("WATCH_LATER_VIDEO_ADDED",a)};
k.Na=function(a){Td("WATCH_LATER_VIDEO_REMOVED",a)};
k.Ja=function(){if(!this.da){if(Db){var a=document,b=a.scrollingElement?a.scrollingElement:jb||"CSS1Compat"!=a.compatMode?a.body||a.documentElement:a.documentElement,a=a.parentWindow||a.defaultView;this.ba=G&&wb("10")&&a.pageYOffset!=b.scrollTop?new Ha(b.scrollLeft,b.scrollTop):new Ha(a.pageXOffset||b.scrollLeft,a.pageYOffset||b.scrollTop);Tf(this.I,window,"scroll",this.Qa);Tf(this.I,this.j,"touchmove",this.Pa)}else Tf(this.I,this.j,"mousewheel",this.qa),Tf(this.I,this.j,"wheel",this.qa);this.da=
!0}};
k.Ka=function(){Uf(this.I);this.da=!1};
k.qa=function(a){a=a||window.event;a.returnValue=!1;a.preventDefault&&a.preventDefault()};
k.Qa=function(){window.scrollTo(this.ba.b,this.ba.f)};
k.Pa=function(a){a.preventDefault()};
k.Ca=function(){return this.G||gh(this)};
k.Ba=function(){return this.ca};
function gh(a){return(a=eh(a))?"div"==a.tagName.toLowerCase()?"html5":"flash":null}
function hh(a){Jg("dcp");a.cancel();Xg(a);a.G=null;a.b&&(a.b.loaded=!1);var b=eh(a);"html5"==gh(a)?fh(a)||!ih(a)?a.m=b:(b&&b.destroy&&b.destroy(),a.m=null):b&&b.destroy&&b.destroy();for(var b=a.j,c;c=b.firstChild;)b.removeChild(c);Uf(a.T);a.w=null;a.H=null}
k.cancel=function(){this.K&&be(this.b.assets.js,this.K);window.clearTimeout(this.la);this.R=!1};
k.l=function(){hh(this);if(this.m&&this.b&&this.m.destroy)try{this.m.destroy()}catch(b){S(b)}this.X=null;for(var a in this.i)p[this.i[a]]=null;this.ea=this.b=this.g=null;delete this.j;delete this.u;Wg.o.l.call(this)};
function ih(a){return a.b&&a.b.args&&a.b.args.fflags?-1!=a.b.args.fflags.indexOf("player_destroy_old_version=true"):!1}
;var lh={},mh="player_uid_"+(1E9*Math.random()>>>0);function nh(a){var b="player",b=r(b)?Sb(b):b;a=Qf(a);var c=mh+"_"+(b[sa]||(b[sa]=++ta)),d=lh[c];if(d)return d.ka(a),d.g;d=new Wg(b,c);lh[c]=d;Td("player-added",d.g);Hc(d,w(oh,d));U(function(){d.ka(a)},0);
return d.g}
function oh(a){lh[a.M]=null}
;function ph(a,b,c){if(ra(a)){b="endSeconds startSeconds mediaContentUrl suggestedQuality videoId two_stage_token".split(" ");c={};for(var d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}return{videoId:a,startSeconds:b,suggestedQuality:c}}
function qh(a,b,c){r(a)&&(a={mediaContentUrl:a,startSeconds:b,suggestedQuality:c});b=/\/([ve]|embed)\/([^#?]+)/.exec(a.mediaContentUrl);a.videoId=b&&b[2]?b[2]:null;return ph(a)}
function rh(a,b,c,d){if(ra(a)&&!oa(a)){b="playlist list listType index startSeconds suggestedQuality".split(" ");c={};for(d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}b={index:b,startSeconds:c,suggestedQuality:d};r(a)&&16==a.length?b.list="PL"+a:b.playlist=a;return b}
function sh(a){var b=a.video_id||a.videoId;if(r(b)){var c=uf()||{},d=uf()||{};q(void 0)?d[b]=void 0:delete d[b];var e=y()+3E5,f=wf;if(f&&window.JSON){r(d)||(d=JSON.stringify(d,void 0));try{f.set("yt-player-two-stage-token",d,e)}catch(g){f.remove("yt-player-two-stage-token")}}(b=c[b])&&(a.two_stage_token=b)}}
function th(a){return(0==a.search("cue")||0==a.search("load"))&&"loadModule"!=a}
;function uh(a){K.call(this);this.g=a;this.g.subscribe("command",this.ta,this);this.h={};this.i=!1}
z(uh,K);k=uh.prototype;k.start=function(){this.i||this.f||(this.i=!0,vh(this.g,"RECEIVING"))};
k.ta=function(a,b){if(this.i&&!this.f){var c=b||{};switch(a){case "addEventListener":if(r(c.event)&&(c=c.event,!(c in this.h))){var d=v(this.Ua,this,c);this.h[c]=d;this.addEventListener(c,d)}break;case "removeEventListener":r(c.event)&&wh(this,c.event);break;default:this.b.isReady()&&this.b[a]&&(c=xh(a,b||{}),c=this.b[a].apply(this.b,c),(c=yh(a,c))&&this.i&&!this.f&&vh(this.g,a,c))}}};
k.Ua=function(a,b){this.i&&!this.f&&vh(this.g,a,this.fa(a,b))};
k.fa=function(a,b){if(null!=b)return{value:b}};
function wh(a,b){b in a.h&&(a.removeEventListener(b,a.h[b]),delete a.h[b])}
k.l=function(){var a=this.g;a.f||pd(a.b,"command",this.ta,this);this.g=null;for(var b in this.h)wh(this,b);uh.o.l.call(this)};function zh(a,b){uh.call(this,b);this.b=a;this.start()}
z(zh,uh);zh.prototype.addEventListener=function(a,b){this.b.addEventListener(a,b)};
zh.prototype.removeEventListener=function(a,b){this.b.removeEventListener(a,b)};
function xh(a,b){switch(a){case "loadVideoById":return b=ph(b),sh(b),[b];case "cueVideoById":return b=ph(b),sh(b),[b];case "loadVideoByPlayerVars":return sh(b),[b];case "cueVideoByPlayerVars":return sh(b),[b];case "loadPlaylist":return b=rh(b),sh(b),[b];case "cuePlaylist":return b=rh(b),sh(b),[b];case "seekTo":return[b.seconds,b.allowSeekAhead];case "playVideoAt":return[b.index];case "setVolume":return[b.volume];case "setPlaybackQuality":return[b.suggestedQuality];case "setPlaybackRate":return[b.suggestedRate];
case "setLoop":return[b.loopPlaylists];case "setShuffle":return[b.shufflePlaylist];case "getOptions":return[b.module];case "getOption":return[b.module,b.option];case "setOption":return[b.module,b.option,b.value];case "handleGlobalKeyDown":return[b.keyCode,b.shiftKey]}return[]}
function yh(a,b){switch(a){case "isMuted":return{muted:b};case "getVolume":return{volume:b};case "getPlaybackRate":return{playbackRate:b};case "getAvailablePlaybackRates":return{availablePlaybackRates:b};case "getVideoLoadedFraction":return{videoLoadedFraction:b};case "getPlayerState":return{playerState:b};case "getCurrentTime":return{currentTime:b};case "getPlaybackQuality":return{playbackQuality:b};case "getAvailableQualityLevels":return{availableQualityLevels:b};case "getDuration":return{duration:b};
case "getVideoUrl":return{videoUrl:b};case "getVideoEmbedCode":return{videoEmbedCode:b};case "getPlaylist":return{playlist:b};case "getPlaylistIndex":return{playlistIndex:b};case "getOptions":return{options:b};case "getOption":return{option:b}}}
zh.prototype.fa=function(a,b){switch(a){case "onReady":return;case "onStateChange":return{playerState:b};case "onPlaybackQualityChange":return{playbackQuality:b};case "onPlaybackRateChange":return{playbackRate:b};case "onError":return{errorCode:b}}return zh.o.fa.call(this,a,b)};
zh.prototype.l=function(){zh.o.l.call(this);delete this.b};function Ah(a,b,c,d){K.call(this);this.g=b||null;this.u="*";this.h=c||null;this.sessionId=null;this.channel=d||null;this.A=!!a;this.m=v(this.w,this);window.addEventListener("message",this.m)}
n(Ah,K);
Ah.prototype.w=function(a){if(!("*"!=this.h&&a.origin!=this.h||this.g&&a.source!=this.g)&&r(a.data)){try{var b=Qc(a.data)}catch(c){return}if(!(null==b||this.A&&(this.sessionId&&this.sessionId!=b.id||this.channel&&this.channel!=b.channel))&&b)switch(b.event){case "listening":"null"!=a.origin?this.h=this.u=a.origin:S(Error("MessageEvent origin is null"),"WARNING");this.g=a.source;this.sessionId=b.id;this.b&&(this.b(),this.b=null);break;case "command":this.i&&(this.j&&!(0<=Ba(this.j,b.func))||this.i(b.func,
b.args))}}};
Ah.prototype.sendMessage=function(a,b){var c=b||this.g;if(c){this.sessionId&&(a.id=this.sessionId);this.channel&&(a.channel=this.channel);try{var d=Rc(a);c.postMessage(d,this.u)}catch(e){S(e,"WARNING")}}};
Ah.prototype.l=function(){window.removeEventListener("message",this.m);K.prototype.l.call(this)};function Bh(a,b,c){Ah.call(this,a,b,c||Q("POST_MESSAGE_ORIGIN",void 0)||window.document.location.protocol+"//"+window.document.location.hostname,"widget");this.j=this.b=this.i=null}
n(Bh,Ah);function Ch(){var a=!!Q("WIDGET_ID_ENFORCE"),a=this.f=new Bh(a),b=v(this.Ra,this);a.i=b;a.j=null;this.f.channel="widget";if(a=Q("WIDGET_ID"))this.f.sessionId=a;this.h=[];this.j=!1;this.i={}}
k=Ch.prototype;k.Ra=function(a,b){if("addEventListener"==a&&b){var c=b[0];this.i[c]||"onReady"==c||(this.addEventListener(c,Dh(this,c)),this.i[c]=!0)}else this.va(a,b)};
k.va=function(){};
function Dh(a,b){return v(function(a){this.sendMessage(b,a)},a)}
k.addEventListener=function(){};
k.za=function(){this.j=!0;this.sendMessage("initialDelivery",this.ga());this.sendMessage("onReady");C(this.h,this.ua,this);this.h=[]};
k.ga=function(){return null};
function Eh(a,b){a.sendMessage("infoDelivery",b)}
k.ua=function(a){this.j?this.f.sendMessage(a):this.h.push(a)};
k.sendMessage=function(a,b){this.ua({event:a,info:void 0==b?null:b})};
k.dispose=function(){this.f=null};function Fh(a){Ch.call(this);this.b=a;this.g=[];this.addEventListener("onReady",v(this.Oa,this));this.addEventListener("onVideoProgress",v(this.Ya,this));this.addEventListener("onVolumeChange",v(this.Za,this));this.addEventListener("onApiChange",v(this.Ta,this));this.addEventListener("onPlaybackQualityChange",v(this.Va,this));this.addEventListener("onPlaybackRateChange",v(this.Wa,this));this.addEventListener("onStateChange",v(this.Xa,this))}
z(Fh,Ch);k=Fh.prototype;k.va=function(a,b){if(this.b[a]){b=b||[];if(0<b.length&&th(a)){var c=b;if(ra(c[0])&&!oa(c[0]))c=c[0];else{var d={};switch(a){case "loadVideoById":case "cueVideoById":d=ph.apply(window,c);break;case "loadVideoByUrl":case "cueVideoByUrl":d=qh.apply(window,c);break;case "loadPlaylist":case "cuePlaylist":d=rh.apply(window,c)}c=d}sh(c);b.length=1;b[0]=c}this.b[a].apply(this.b,b);th(a)&&Eh(this,this.ga())}};
k.Oa=function(){var a=v(this.za,this);this.f.b=a};
k.addEventListener=function(a,b){this.g.push({eventType:a,listener:b});this.b.addEventListener(a,b)};
k.ga=function(){if(!this.b)return null;var a=this.b.getApiInterface();Ea(a,"getVideoData");for(var b={apiInterface:a},c=0,d=a.length;c<d;c++){var e=a[c],f=e;if(0==f.search("get")||0==f.search("is")){var f=e,g=0;0==f.search("get")?g=3:0==f.search("is")&&(g=2);f=f.charAt(g).toLowerCase()+f.substr(g+1);try{var h=this.b[e]();b[f]=h}catch(l){}}}b.videoData=this.b.getVideoData();b.currentTimeLastUpdated_=y()/1E3;return b};
k.Xa=function(a){a={playerState:a,currentTime:this.b.getCurrentTime(),duration:this.b.getDuration(),videoData:this.b.getVideoData(),videoStartBytes:0,videoBytesTotal:this.b.getVideoBytesTotal(),videoLoadedFraction:this.b.getVideoLoadedFraction(),playbackQuality:this.b.getPlaybackQuality(),availableQualityLevels:this.b.getAvailableQualityLevels(),videoUrl:this.b.getVideoUrl(),playlist:this.b.getPlaylist(),playlistIndex:this.b.getPlaylistIndex(),currentTimeLastUpdated_:y()/1E3,playbackRate:this.b.getPlaybackRate(),
mediaReferenceTime:this.b.getMediaReferenceTime()};this.b.getProgressState&&(a.progressState=this.b.getProgressState());this.b.getStoryboardFormat&&(a.storyboardFormat=this.b.getStoryboardFormat());Eh(this,a)};
k.Va=function(a){Eh(this,{playbackQuality:a})};
k.Wa=function(a){Eh(this,{playbackRate:a})};
k.Ta=function(){for(var a=this.b.getOptions(),b={namespaces:a},c=0,d=a.length;c<d;c++){var e=a[c],f=this.b.getOptions(e);b[e]={options:f};for(var g=0,h=f.length;g<h;g++){var l=f[g],m=this.b.getOption(e,l);b[e][l]=m}}this.sendMessage("apiInfoDelivery",b)};
k.Za=function(){Eh(this,{muted:this.b.isMuted(),volume:this.b.getVolume()})};
k.Ya=function(a){a={currentTime:a,videoBytesLoaded:this.b.getVideoBytesLoaded(),videoLoadedFraction:this.b.getVideoLoadedFraction(),currentTimeLastUpdated_:y()/1E3,playbackRate:this.b.getPlaybackRate(),mediaReferenceTime:this.b.getMediaReferenceTime()};this.b.getProgressState&&(a.progressState=this.b.getProgressState());Eh(this,a)};
k.dispose=function(){Fh.o.dispose.call(this);for(var a=0;a<this.g.length;a++){var b=this.g[a];this.b.removeEventListener(b.eventType,b.listener)}this.g=[]};function Gh(){K.call(this);this.b=new M;Hc(this,w(Ic,this.b))}
z(Gh,K);Gh.prototype.subscribe=function(a,b,c){return this.f?0:this.b.subscribe(a,b,c)};
Gh.prototype.D=function(a){return this.f?!1:this.b.D(a)};
Gh.prototype.j=function(a,b){this.f||this.b.O.apply(this.b,arguments)};function Hh(a,b,c){Gh.call(this);this.g=a;this.h=b;this.i=c}
z(Hh,Gh);function vh(a,b,c){if(!a.f){var d=a.g;d.f||a.h!=d.b||(a={id:a.i,command:b},c&&(a.data=c),d.b.postMessage(Rc(a),d.h))}}
Hh.prototype.l=function(){this.h=this.g=null;Hh.o.l.call(this)};function Ih(a,b,c){K.call(this);this.b=a;this.h=c;this.i=Te(window,"message",v(this.j,this));this.g=new Hh(this,a,b);Hc(this,w(Ic,this.g))}
z(Ih,K);Ih.prototype.j=function(a){var b;if(b=!this.f)if(b=a.origin==this.h)a:{b=this.b;do{b:{var c=a.source;do{if(c==b){c=!0;break b}if(c==c.parent)break;c=c.parent}while(null!=c);c=!1}if(c){b=!0;break a}b=b.opener}while(null!=b);b=!1}if(b&&(a=a.data,r(a))){try{a=Qc(a)}catch(d){return}a.command&&(b=this.g,b.f||b.j("command",a.command,a.data))}};
Ih.prototype.l=function(){Ue(this.i);this.b=null;Ih.o.l.call(this)};function Jh(){var a=Kh("html5_serverside_pagead_id_sets_cookie","EXP_HTML5_SERVERSIDE_PAGEAD_ID_SETS_COOKIE")?"//googleads.g.doubleclick.net/pagead/id?exp=nomnom":"//googleads.g.doubleclick.net/pagead/id",b=Oa(Lh);return new L(function(c,d){b.P=function(a){ke(a)?c(a):d(new Mh("Request failed, status="+a.status,"net.badstatus"))};
b.onError=function(){d(new Mh("Unknown request error","net.unknown"))};
b.aa=function(){d(new Mh("Request timed out","net.timeout"))};
re(a,b)})}
function Mh(a,b){B.call(this,a+", errorCode="+b);this.errorCode=b;this.name="PromiseAjaxError"}
n(Mh,B);function Nh(a){this.g=void 0===a?null:a;this.f=0;this.b=null}
Nh.prototype.then=function(a,b,c){return this.g?this.g.then(a,b,c):1===this.f&&a?(a=a.call(c,this.b),Yc(a)?a:Oh(a)):2===this.f&&b?(a=b.call(c,this.b),Yc(a)?a:Ph(a)):this};
Nh.prototype.getValue=function(){return this.b};
Xc(Nh);function Ph(a){var b=new Nh;a=void 0===a?null:a;b.f=2;b.b=void 0===a?null:a;return b}
function Oh(a){var b=new Nh;a=void 0===a?null:a;b.f=1;b.b=void 0===a?null:a;return b}
;function Qh(a){B.call(this,a.message||a.description||a.name)}
n(Qh,B);Qh.prototype.name="BiscottiError";function Rh(){B.call(this,"Biscotti ID is missing from server")}
n(Rh,B);Rh.prototype.name="BiscottiMissingError";var Lh={format:"RAW",method:"GET",timeout:5E3,withCredentials:!0},Sh=null;function Th(){if("1"===Ja(Q("PLAYER_CONFIG",{}),"args","privembed"))return cd(Error("Biscotti ID is not available in private embed mode"));Sh||(Sh=ed(Jh().then(Uh),function(a){return Vh(2,a)}));
return Sh}
function Kh(a,b){return!!Ja(window,"settings","experiments","flags",a)||!!Q(b,!1)||!!T(a)}
function Uh(a){a=a.responseText;if(0!=a.lastIndexOf(")]}'",0))throw new Rh;a=JSON.parse(a.substr(4));if(Kh("html5_serverside_ignore_biscotti_id_on_retry","EXP_HTML5_SERVERSIDE_IGNORE_BISCOTTI_ID_ON_RETRY")&&1<(a.type||1))throw new Rh;a=a.id;Wh(a);Sh=Oh(a);Xh(18E5,2);return a}
function Vh(a,b){var c=new Qh(b);Wh("");Sh=Ph(c);0<a&&Xh(12E4,a-1);throw c;}
function Xh(a,b){U(function(){ed(Jh().then(Uh,function(a){return Vh(b,a)}),la)},a)}
function Wh(a){t("yt.ads.biscotti.lastId_",a,void 0)}
;function Yh(){}
function Zh(a){if("1"!==Ja(Q("PLAYER_CONFIG",{}),"args","privembed")){a&&!u("yt.ads.biscotti.getId_")&&t("yt.ads.biscotti.getId_",Th,void 0);try{try{var b=u("yt.ads.biscotti.getId_");var c=b?b():Th()}catch(d){c=cd(d)}c.then($h,Yh);U(Zh,18E5)}catch(d){S(d)}}}
var ai=0;
function $h(a){a:{try{var b=window.top.location.href}catch(Ra){b=2;break a}b=null!=b?b==window.document.location.href?0:1:2}b={dt:lc,flash:bb||"0",frm:b};b.u_tz=-(new Date).getTimezoneOffset();var c=void 0===c?A:c;try{var d=c.history.length}catch(Ra){d=0}b.u_his=d;b.u_java=!!A.navigator&&"unknown"!==typeof A.navigator.javaEnabled&&!!A.navigator.javaEnabled&&A.navigator.javaEnabled();A.screen&&(b.u_h=A.screen.height,b.u_w=A.screen.width,b.u_ah=A.screen.availHeight,b.u_aw=A.screen.availWidth,b.u_cd=
A.screen.colorDepth);A.navigator&&A.navigator.plugins&&(b.u_nplug=A.navigator.plugins.length);A.navigator&&A.navigator.mimeTypes&&(b.u_nmime=A.navigator.mimeTypes.length);b.bid=a;b.ca_type=ab?"flash":"image";if(T("enable_server_side_search_pyv")||T("enable_server_side_mweb_search_pyv")){a=window;try{var e=a.screenX;var f=a.screenY}catch(Ra){}try{var g=a.outerWidth;var h=a.outerHeight}catch(Ra){}try{var l=a.innerWidth;var m=a.innerHeight}catch(Ra){}e=[a.screenLeft,a.screenTop,e,f,a.screen?a.screen.availWidth:
void 0,a.screen?a.screen.availTop:void 0,g,h,l,m];f=window.top||A;try{if(f.document&&!f.document.body)var x=new Ia(-1,-1);else{var D=(f||window).document,Hd="CSS1Compat"==D.compatMode?D.documentElement:D.body;x=(new Ia(Hd.clientWidth,Hd.clientHeight)).round()}var za=x}catch(Ra){za=new Ia(-12245933,-12245933)}x=0;window.SVGElement&&document.createElementNS&&(x|=1);za={bc:x,bih:za.height,biw:za.width,brdim:e.join(),vis:{visible:1,hidden:2,prerender:3,preview:4}[wa.webkitVisibilityState||wa.mozVisibilityState||
wa.visibilityState||""]||0,wgl:!!A.WebGLRenderingContext};for(var Id in za)b[Id]=za[Id]}b.bsq=ai++;xe("//www.youtube.com/ad_data_204",{Da:!1,B:b})}
;function bi(){this.b=Q("ALT_PREF_COOKIE_NAME","PREF");var a=rc.get(""+this.b,void 0);if(a)for(var a=unescape(a).split("&"),b=0;b<a.length;b++){var c=a[b].split("="),d=c[0];(c=c[1])&&(Z[d]=c.toString())}}
ma(bi);var Z=u("yt.prefs.UserPrefs.prefs_")||{};t("yt.prefs.UserPrefs.prefs_",Z,void 0);function ci(a){if(/^f([1-9][0-9]*)$/.test(a))throw Error("ExpectedRegexMatch: "+a);}
function di(a){if(!/^\w+$/.test(a))throw Error("ExpectedRegexMismatch: "+a);}
function ei(a){a=void 0!==Z[a]?Z[a].toString():null;return null!=a&&/^[A-Fa-f0-9]+$/.test(a)?parseInt(a,16):null}
bi.prototype.get=function(a,b){di(a);ci(a);var c=void 0!==Z[a]?Z[a].toString():null;return null!=c?c:b?b:""};
bi.prototype.set=function(a,b){di(a);ci(a);if(null==b)throw Error("ExpectedNotNull");Z[a]=b.toString()};
bi.prototype.remove=function(a){di(a);ci(a);delete Z[a]};
bi.prototype.clear=function(){Z={}};var fi=null,gi=null,hi=null,ii={};function ji(a){of(a.payload_name,a.payload,rf,void 0,void 0)}
function ki(a){var b=a.id;a=a.ve_type;var c=Ze++;a=new Ye(void 0,a,c,void 0,void 0);ii[b]=a;b=Af();c=zf();b&&c&&pf(b,c,a)}
function li(a){var b=a.csn;a=a.root_ve_type;if(b&&a&&(P("client-screen-nonce",b),P("ROOT_VE_TYPE",a),a=zf()))for(var c in ii){var d=ii[c];d&&pf(b,a,d)}}
function mi(a){ii[a.id]=new Ye(a.tracking_params)}
function ni(a){var b=Af();a=ii[a.id];b&&a&&qf(rf,{click:{csn:b,visualElement:$e(a)}},void 0,void 0)}
function oi(a){a=a.ids;var b=Af();if(b){for(var c=[],d=0;d<a.length;d++){var e=ii[a[d]];e&&c.push(e)}c.length&&sf(b,c)}}
function pi(){var a=fi;a&&a.startInteractionLogging&&a.startInteractionLogging()}
;t("yt.setConfig",P,void 0);t("yt.config.set",P,void 0);t("yt.setMsg",Ge,void 0);t("yt.msgs.set",Ge,void 0);t("yt.logging.errors.log",Ce,void 0);
t("writeEmbed",function(){var a=Q("PLAYER_CONFIG",void 0);Zh(!0);"gvn"==a.args.ps&&(document.body.style.backgroundColor="transparent");var b=document.referrer,c=Q("POST_MESSAGE_ORIGIN");window!=window.top&&b&&b!=document.URL&&(a.args.loaderUrl=b);Q("LIGHTWEIGHT_AUTOPLAY")&&(a.args.autoplay="1");a.args.autoplay&&sh(a.args);fi=a=nh(a);a.addEventListener("onScreenChanged",li);a.addEventListener("onLogClientVeCreated",ki);a.addEventListener("onLogServerVeCreated",mi);a.addEventListener("onLogToGel",ji);
a.addEventListener("onLogVeClicked",ni);a.addEventListener("onLogVesShown",oi);a.addEventListener("onReady",pi);b=Q("POST_MESSAGE_ID","player");Q("ENABLE_JS_API")?hi=new Fh(a):Q("ENABLE_POST_API")&&r(b)&&r(c)&&(gi=new Ih(window.parent,b,c),hi=new zh(a,gi.g));Q("BG_P")&&(Q("BG_I")||Q("BG_IU"))&&ee();Le()},void 0);
t("yt.www.watch.ads.restrictioncookie.spr",function(a){Ae(a+"mac_204?action_fcts=1");return!0},void 0);
var qi=Ed(function(){Jg("ol");var a=bi.getInstance(),b=1<window.devicePixelRatio;if(!!((ei("f"+(Math.floor(119/31)+1))||0)&67108864)!=b){var c="f"+(Math.floor(119/31)+1),d=ei(c)||0,d=b?d|67108864:d&-67108865;0==d?delete Z[c]:Z[c]=d.toString(16).toString();var a=a.b,b=[],e;for(e in Z)b.push(e+"="+escape(Z[e]));Bf(a,b.join("&"),63072E3)}}),ri=Ed(function(){var a=fi;
a&&a.sendAbandonmentPing&&a.sendAbandonmentPing();Q("PL_ATT")&&(de=null);for(var a=0,b=Je.length;a<b;a++)Ie(Je[a]);Je.length=0;ae("//static.doubleclick.net/instream/ad_status.js");Ke=!1;P("DCLKSTAT",0);Jc(hi,gi);if(a=fi)a.removeEventListener("onScreenChanged",li),a.removeEventListener("onLogClientVeCreated",ki),a.removeEventListener("onLogServerVeCreated",mi),a.removeEventListener("onLogToGel",ji),a.removeEventListener("onLogVeClicked",ni),a.removeEventListener("onLogVesShown",oi),a.removeEventListener("onReady",
pi),a.destroy();ii={}});
window.addEventListener?(window.addEventListener("load",qi),window.addEventListener("unload",ri)):window.attachEvent&&(window.attachEvent("onload",qi),window.attachEvent("onunload",ri));}).call(this);
