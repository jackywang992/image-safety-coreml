import Foundation
struct ImageSafetyBloodContextResiduals {
 enum Failure: Error {case invalidModel,invalidFeatures}
 let n:Int;let gamma:Double;let minNormal:Double;let margin:Double;let support:[Float];let coefficient:[Float];let xsCoefficient:[Float];let risk:[Bool]
 init(data:Data)throws {
  guard data.count>=32,Array(data.prefix(8))==Array("MQBG3601".utf8) else {throw Failure.invalidModel}
  func u32(_ at:Int)->UInt32 {data.withUnsafeBytes{UInt32(littleEndian:$0.loadUnaligned(fromByteOffset:at,as:UInt32.self))}}
  n=Int(u32(8));guard n>0,n<=8192,u32(12)==768,u32(28)==0,data.count==32+n*3081 else {throw Failure.invalidModel}
  gamma=Double(Float(bitPattern:u32(16)));minNormal=Double(Float(bitPattern:u32(20)));margin=Double(Float(bitPattern:u32(24)))
  guard gamma.isFinite,gamma>0,gamma<=32,minNormal.isFinite,minNormal>=0,minNormal<=1,margin.isFinite,margin>=0,margin<=1 else {throw Failure.invalidModel}
  let count=n
  support=(0..<(count*768)).map{Float(bitPattern:u32(32+4*$0))};coefficient=(0..<count).map{Float(bitPattern:u32(32+4*count*768+4*$0))}
  xsCoefficient=(0..<count).map{Float(bitPattern:u32(32+4*count*768+4*count+4*$0))}
  let flags=Array(data.suffix(n));guard flags.allSatisfy({$0<=1}),flags.contains(0),flags.contains(1),support.allSatisfy({$0.isFinite}),coefficient.allSatisfy({$0.isFinite}),xsCoefficient.allSatisfy({$0.isFinite}) else {throw Failure.invalidModel};risk=flags.map{$0==1}
  for i in 0..<n {var norm=0.0;for j in 0..<768 {let v=Double(support[i*768+j]);norm+=v*v};guard abs(norm-1)<0.001 else {throw Failure.invalidModel}}
 }
 func recallProbability(whole:[Float],center:[Float])throws->Float {try recallEvidence(whole:whole,center:center).gated}
 func recallEvidence(whole:[Float],center:[Float])throws->(gated:Float,raw:Float) {
  guard whole.count>=384,center.count>=384 else {throw Failure.invalidFeatures}
  var x=(Array(whole.prefix(384))+Array(center.prefix(384))).map(Double.init)
  let norm=sqrt(x.reduce(0){$0+$1*$1});guard norm.isFinite,norm>0 else {throw Failure.invalidFeatures}
  for j in 0..<768{x[j]/=norm}
  var target = -Double.infinity,counterexample = -Double.infinity,value=0.0
  for i in 0..<n {var dot=0.0;for j in 0..<768{dot+=x[j]*Double(support[i*768+j])}
   if risk[i]{counterexample=max(counterexample,dot)}else{target=max(target,dot)}
   value+=exp(-gamma*max(0,2-2*dot))*Double(coefficient[i])
  }
  guard value.isFinite else {throw Failure.invalidFeatures}
  let raw=Float(1/(1+exp(-value)))
  if target<minNormal || target<counterexample+margin{return (0,raw)}
  return (raw,raw)
 }
 func correction(whole:[Float],center:[Float])throws->(gore:Float,xs:Float) {
  guard whole.count>=384,center.count>=384 else {throw Failure.invalidFeatures}
  var x=(Array(whole.prefix(384))+Array(center.prefix(384))).map(Double.init);let norm=sqrt(x.reduce(0){$0+$1*$1});guard norm.isFinite,norm>0 else {throw Failure.invalidFeatures};for j in 0..<768{x[j]/=norm}
  var normal = -Double.infinity,unsafe = -Double.infinity,value=0.0,xsValue=0.0
  for i in 0..<n {var dot=0.0;for j in 0..<768{dot+=x[j]*Double(support[i*768+j])};if risk[i]{unsafe=max(unsafe,dot)}else{normal=max(normal,dot)};let k=exp(-gamma*max(0,2-2*dot));value+=k*Double(coefficient[i]);xsValue+=k*Double(xsCoefficient[i])}
  guard value.isFinite,xsValue.isFinite else {throw Failure.invalidFeatures};if normal<minNormal || normal<unsafe+margin{return (0,0)};let result=Float(max(0,value)),xsResult=Float(max(0,xsValue));guard result.isFinite,xsResult.isFinite else {throw Failure.invalidFeatures};return (result,xsResult)
 }
}
