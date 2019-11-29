var fs = require("fs");
var datetime = require('node-datetime');
var dt = datetime.create();
var formatted = dt.format('dmY_HMS');
var anno = JSON.parse(fs.readFileSync("images-p1-221119_AT1_via205_221119.json"));
var list = fs.readFileSync("nopred.json",'utf8');
//console.log(list);
console.log({list:list.length});
var listA = list.split(/\n/);
//console.log(listA);
console.log({listA:listA.length});
var listH = {};
for (var one of listA) {
  var k = one.split(/\//)
  var key = k[k.length-1];
  listH[key] = true;
}
//console.log(listH);
console.log({listH:Object.keys(listH).length});

console.log(anno);
var ret = {};
for (var one in anno) {
  console.log(one);
  var key = one.replace(/[\-0-9][0-9]*$/,'');
  console.log(key);
  if (!listH[key]) { continue; }
  ret[one] = anno[one];
}
console.log(ret);
console.log({ret:Object.keys(ret).length});
fs.writeFileSync("images-p1-221119_AT1_via205_221119_filtered_"+formatted+".json",JSON.stringify(ret));

